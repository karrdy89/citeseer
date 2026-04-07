"""Semantic Scholar API 클라이언트 — retry queue + prefetch."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx

from app.core.config import Settings
from app.s2.models import s2_to_paper

logger = logging.getLogger(__name__)

_BACKOFF_BASE = 2.0  # 재시도 대기: 2s, 4s, 8s …


@dataclass
class FetchTask:
    token: str | None
    page: int
    retries: int = 0


class S2Fetcher:
    def __init__(self, settings: Settings, client: httpx.AsyncClient, output_queue: asyncio.Queue):
        self.settings = settings
        self.client = client
        self.output_queue = output_queue
        self._retry_queue: asyncio.Queue[FetchTask] = asyncio.Queue()
        self._request_count = 0
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

    @property
    def request_count(self) -> int:
        return self._request_count

    def _at_limit(self) -> bool:
        return self.settings.s2_max_requests > 0 and self._request_count >= self.settings.s2_max_requests

    async def run(self) -> None:
        try:
            await self._fetch_loop()
        finally:
            self._done = True
            await self.output_queue.put(None)

    async def _fetch_loop(self) -> None:
        s = self.settings
        base_params = {
            "query": "*",
            "venue": ",".join(s.target_venues),
            "publicationDateOrYear": f"{s.year_from}:{s.year_to}",
            "fields": s.s2_fields,
            "sort": "publicationDate:desc",
        }
        headers = {"x-api-key": s.s2_api_key} if s.s2_api_key else {}
        current = FetchTask(token=None, page=0)

        while True:
            if self._at_limit():
                logger.info(f"S2 request limit reached ({self._request_count}/{s.s2_max_requests})")
                break

            if not self._retry_queue.empty():
                current = self._retry_queue.get_nowait()
                backoff = _BACKOFF_BASE ** current.retries
                logger.info(f"Retrying page {current.page} (attempt {current.retries + 1}, backoff {backoff:.0f}s)")
                await asyncio.sleep(backoff)

            params = dict(base_params)
            if current.token:
                params["token"] = current.token

            try:
                await asyncio.sleep(s.s2_rate_limit)
                resp = await self.client.get(
                    f"{s.s2_base_url}/paper/search/bulk", params=params, headers=headers, timeout=s.s2_timeout,
                )
                self._request_count += 1
                resp.raise_for_status()
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                logger.warning(f"S2 fetch failed (page {current.page}): {e}")
                if current.retries < s.max_retries:
                    current.retries += 1
                    await self._retry_queue.put(current)
                    continue
                else:
                    logger.error(f"S2 page {current.page} failed after {current.retries} retries, stopping")
                    break

            body = resp.json()
            papers_data = body.get("data") or []
            if not papers_data:
                logger.info("No more papers from S2.")
                break

            papers = [s2_to_paper(p) for p in papers_data]
            logger.info(f"Fetched page {current.page}: {len(papers)} papers")
            await self.output_queue.put(papers)

            next_token = body.get("token")
            if not next_token:
                break
            current = FetchTask(token=next_token, page=current.page + 1)
