"""Elasticsearch 클라이언트 — 재시도 + backoff."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

import httpx

from app.core.config import Settings
from app.es.mapping import PAPERS_MAPPING

logger = logging.getLogger(__name__)

_BACKOFF_BASE = 2.0


async def _with_retry(label: str, settings: Settings, coro_fn):
    """공통 재시도 래퍼."""
    for attempt in range(settings.max_retries):
        try:
            return await coro_fn()
        except httpx.HTTPError as e:
            if attempt < settings.max_retries - 1:
                backoff = _BACKOFF_BASE ** (attempt + 1)
                logger.warning(f"{label} failed (attempt {attempt + 1}): {e}, retrying in {backoff:.0f}s")
                await asyncio.sleep(backoff)
            else:
                raise


async def ensure_index(settings: Settings, client: httpx.AsyncClient) -> None:
    async def _do():
        resp = await client.head(f"{settings.es_host}/{settings.es_index}", timeout=settings.es_timeout)
        if resp.status_code == 200:
            logger.info(f"Index '{settings.es_index}' already exists.")
            return
        resp = await client.put(
            f"{settings.es_host}/{settings.es_index}", json=PAPERS_MAPPING, timeout=settings.es_timeout,
        )
        resp.raise_for_status()
        logger.info(f"Index '{settings.es_index}' created.")

    await _with_retry("ensure_index", settings, _do)


async def get_existing_ids(settings: Settings, client: httpx.AsyncClient) -> set[str]:
    ids: set[str] = set()
    body = {"size": 10000, "query": {"match_all": {}}, "_source": False, "fields": ["s2_id"]}
    scroll_id = None

    try:
        resp = await client.post(
            f"{settings.es_host}/{settings.es_index}/_search?scroll=2m",
            json=body, timeout=settings.es_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        scroll_id = data.get("_scroll_id")
        hits = data.get("hits", {}).get("hits", [])

        while hits:
            for hit in hits:
                s2_ids = hit.get("fields", {}).get("s2_id", [])
                if s2_ids:
                    ids.add(s2_ids[0])
            resp = await client.post(
                f"{settings.es_host}/_search/scroll",
                json={"scroll": "2m", "scroll_id": scroll_id}, timeout=settings.es_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            scroll_id = data.get("_scroll_id")
            hits = data.get("hits", {}).get("hits", [])
    finally:
        if scroll_id:
            try:
                await client.request(
                    "DELETE", f"{settings.es_host}/_search/scroll",
                    json={"scroll_id": scroll_id}, timeout=settings.es_timeout,
                )
            except httpx.HTTPError:
                pass

    logger.info(f"Existing papers in ES: {len(ids)}")
    return ids


def _to_es_doc(paper: dict) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    doc = {
        "doi": paper.get("doi"),
        "arxiv_id": paper.get("arxiv_id"),
        "s2_id": paper["s2_id"],
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        "authors": paper.get("authors", []),
        "year": paper.get("year"),
        "publication_date": paper.get("publication_date"),
        "venue": paper.get("venue"),
        "citation_count": paper.get("citation_count", 0),
        "tldr": paper.get("tldr"),
        "url": paper.get("url"),
        "content_type": paper.get("content_type"),
        "pdf_path": paper.get("pdf_path"),
        "json_path": paper.get("json_path"),
        "ingested_at": now,
        "updated_at": now,
    }
    if paper.get("embedding"):
        doc["embedding"] = paper["embedding"]
    return doc


async def bulk_upsert(
    papers: list[dict], settings: Settings, client: httpx.AsyncClient,
) -> int:
    if not papers:
        return 0

    async def _do():
        lines: list[str] = []
        for paper in papers:
            lines.append(json.dumps({"index": {"_index": settings.es_index, "_id": paper["s2_id"]}}))
            lines.append(json.dumps(_to_es_doc(paper)))
        body = "\n".join(lines) + "\n"

        resp = await client.post(
            f"{settings.es_host}/_bulk", content=body,
            headers={"Content-Type": "application/x-ndjson"}, timeout=settings.es_timeout,
        )
        resp.raise_for_status()
        result = resp.json()
        errors = sum(1 for item in result.get("items", []) if item.get("index", {}).get("error"))
        success = len(papers) - errors
        if errors:
            logger.warning(f"Bulk upsert: {success} ok, {errors} errors")
        else:
            logger.info(f"Bulk upsert: {success} papers")
        return success

    return await _with_retry("bulk_upsert", settings, _do)
