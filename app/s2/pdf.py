"""PDF 다운로드 모듈 — 재시도 + backoff."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)

_BACKOFF_BASE = 2.0


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def get_pdf_url(paper: dict) -> str | None:
    """PDF URL을 구한다. 우선순위: S2 OA → arXiv → ACL Anthology."""
    if paper.get("_s2_oa_url"):
        return paper["_s2_oa_url"]
    if paper.get("arxiv_id"):
        return f"https://arxiv.org/pdf/{paper['arxiv_id']}.pdf"
    if paper.get("doi") and "aclweb" in paper["doi"]:
        slug = paper["doi"].split("/")[-1]
        return f"https://aclanthology.org/{slug}.pdf"
    return None


def _make_pdf_path(paper: dict, settings: Settings) -> str:
    venue = _sanitize(paper.get("venue") or "unknown")
    year = paper.get("year") or "unknown"
    if paper.get("arxiv_id"):
        filename = paper["arxiv_id"]
    elif paper.get("doi"):
        filename = hashlib.sha256(paper["doi"].encode()).hexdigest()[:16]
    else:
        filename = hashlib.sha256(paper["s2_id"].encode()).hexdigest()[:16]
    return os.path.join(settings.pdf_dir, venue, str(year), f"{filename}.pdf")


async def download_pdf(
    paper: dict,
    settings: Settings,
    client: httpx.AsyncClient,
) -> str | None:
    url = get_pdf_url(paper)
    if not url:
        return None
    path = _make_pdf_path(paper, settings)
    if os.path.exists(path):
        logger.debug(f"PDF already exists: {path}")
        return path

    for attempt in range(settings.max_retries):
        try:
            resp = await client.get(url, follow_redirects=True, timeout=settings.pdf_timeout)
            if resp.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"Server error {resp.status_code}", request=resp.request, response=resp,
                )
            if resp.status_code != 200:
                logger.warning(f"PDF download failed ({resp.status_code}): {url}")
                return None  # 4xx → 포기
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and "octet-stream" not in content_type:
                logger.warning(f"Not a PDF ({content_type}): {url}")
                return None
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(resp.content)
            logger.info(f"Downloaded: {path}")
            return path
        except (httpx.HTTPError, OSError) as e:
            if attempt < settings.max_retries - 1:
                backoff = _BACKOFF_BASE ** (attempt + 1)
                logger.warning(f"PDF download error (attempt {attempt + 1}): {url} → {e}, retrying in {backoff:.0f}s")
                await asyncio.sleep(backoff)
            else:
                logger.warning(f"PDF download failed after {settings.max_retries} attempts: {url} → {e}")
                return None
    return None
