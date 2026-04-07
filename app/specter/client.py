"""SPECTER2 임베딩 서비스 클라이언트 — 배치 처리 + 재시도."""
from __future__ import annotations

import asyncio
import logging

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)

_BACKOFF_BASE = 2.0


async def embed_batch(
    texts: list[str],
    settings: Settings,
    client: httpx.AsyncClient,
) -> list[list[float]]:
    if not texts:
        return []
    for attempt in range(settings.max_retries):
        try:
            resp = await client.post(
                f"{settings.specter_url}/encode",
                json={"text": texts},
                timeout=settings.specter_timeout,
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]
        except (httpx.HTTPError, KeyError) as e:
            if attempt < settings.max_retries - 1:
                backoff = _BACKOFF_BASE ** (attempt + 1)
                logger.warning(f"Embedding request failed (attempt {attempt + 1}): {e}, retrying in {backoff:.0f}s")
                await asyncio.sleep(backoff)
            else:
                logger.error(f"Embedding request failed after {settings.max_retries} attempts: {e}")
                return []
    return []  # unreachable, but satisfies type checker


async def embed_papers(
    papers: list[dict],
    settings: Settings,
    client: httpx.AsyncClient,
) -> None:
    """S2 embedding이 없는 논문에 대해 SPECTER2로 임베딩 생성. in-place 업데이트."""
    need_embed = [p for p in papers if not p.get("embedding")]
    if not need_embed:
        return

    batch_size = settings.embedding_batch_size
    for i in range(0, len(need_embed), batch_size):
        batch = need_embed[i:i + batch_size]
        texts = [f"{p['title']} [SEP] {p.get('abstract', '')}" for p in batch]
        vectors = await embed_batch(texts, settings, client)
        if len(vectors) == len(batch):
            for paper, vec in zip(batch, vectors):
                paper["embedding"] = vec
        else:
            logger.warning(f"Embedding count mismatch: expected {len(batch)}, got {len(vectors)}")
