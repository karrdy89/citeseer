"""수집 파이프라인 — 프리페치 + 병렬 sub-batch 처리."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone

import httpx

from app.core.config import Settings, get_settings
from app.core.logging import setup_logging
from app.es.client import bulk_upsert, ensure_index, get_existing_ids
from app.s2.client import S2Fetcher
from app.s2.pdf import download_pdf
from app.specter.client import embed_papers

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 2


def _load_checkpoint(settings: Settings) -> dict:
    if os.path.exists(settings.checkpoint_path):
        with open(settings.checkpoint_path) as f:
            return json.load(f)
    return {}


def _save_checkpoint(settings: Settings, data: dict) -> None:
    os.makedirs(os.path.dirname(settings.checkpoint_path), exist_ok=True)
    with open(settings.checkpoint_path, "w") as f:
        json.dump(data, f, indent=2)


def _apply_checkpoint(checkpoint: dict, settings: Settings) -> None:
    """체크포인트에서 resume 지점 복원."""
    version = checkpoint.get("version", 1)
    if version >= 2 and checkpoint.get("last_collected_date"):
        settings.year_from = checkpoint["last_collected_date"]
        logger.info(f"Resuming from checkpoint: year_from={settings.year_from}")
    elif checkpoint.get("year_from"):
        # 레거시 v1 체크포인트 호환 (int → str)
        settings.year_from = f"{checkpoint['year_from']}-01-01"
        logger.info(f"Resuming from legacy checkpoint: year_from={settings.year_from}")


def _apply_window(settings: Settings) -> None:
    """배치 윈도우 적용 — year_from + window_days로 year_to 제한."""
    if settings.ingest_window_days <= 0:
        return
    start = date.fromisoformat(settings.year_from)
    window_end = start + timedelta(days=settings.ingest_window_days)
    today = date.today()
    if window_end > today:
        window_end = today
    settings.year_to = window_end.isoformat()
    logger.info(f"Window applied: {settings.year_from} ~ {settings.year_to}")


def _oldest_date_in_page(papers: list[dict]) -> str | None:
    """페이지 내 논문 중 가장 오래된 publication_date 반환."""
    dates = [p["publication_date"] for p in papers if p.get("publication_date")]
    return min(dates) if dates else None


def _split_batches(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


async def _process_sub_batch(
    papers: list[dict], settings: Settings, http: httpx.AsyncClient, pdf_sem: asyncio.Semaphore,
) -> int:
    async def _download_all() -> None:
        async def _dl(paper: dict) -> None:
            async with pdf_sem:
                path = await download_pdf(paper, settings, http)
                if path:
                    paper["pdf_path"] = path
                    paper["content_type"] = "pdf"
                else:
                    paper["content_type"] = "abstract_only"
        await asyncio.gather(*[_dl(p) for p in papers])

    await asyncio.gather(embed_papers(papers, settings, http), _download_all())
    return await bulk_upsert(papers, settings, http)


async def _process_page(
    papers: list[dict], existing_ids: set[str], settings: Settings,
    http: httpx.AsyncClient, pdf_sem: asyncio.Semaphore, worker_sem: asyncio.Semaphore,
) -> dict:
    new_papers = [p for p in papers if p["s2_id"] not in existing_ids]
    skipped = len(papers) - len(new_papers)
    if not new_papers:
        return {"new": 0, "indexed": 0, "skipped": skipped}

    sub_batches = _split_batches(new_papers, settings.sub_batch_size)

    async def _run_sub(batch: list[dict]) -> int:
        async with worker_sem:
            return await _process_sub_batch(batch, settings, http, pdf_sem)

    results = await asyncio.gather(*[_run_sub(sb) for sb in sub_batches])
    indexed = sum(results)
    for p in new_papers:
        existing_ids.add(p["s2_id"])
    return {"new": len(new_papers), "indexed": indexed, "skipped": skipped}


async def ingest(settings: Settings | None = None, client: httpx.AsyncClient | None = None) -> dict:
    if settings is None:
        settings = get_settings()

    checkpoint = _load_checkpoint(settings)
    _apply_checkpoint(checkpoint, settings)
    _apply_window(settings)

    stats = {"total_fetched": 0, "new_papers": 0, "indexed": 0, "s2_requests": 0}
    oldest_date: str | None = None
    prefetch_queue: asyncio.Queue = asyncio.Queue(maxsize=settings.prefetch_buffer)
    pdf_sem = asyncio.Semaphore(settings.pdf_concurrency)
    worker_sem = asyncio.Semaphore(settings.worker_concurrency)

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient()

    try:
        http = client
        await ensure_index(settings, http)
        existing_ids = await get_existing_ids(settings, http)

        fetcher = S2Fetcher(settings, http, prefetch_queue)
        fetch_task = asyncio.create_task(fetcher.run())

        while True:
            page_papers = await prefetch_queue.get()
            if page_papers is None:
                break
            stats["total_fetched"] += len(page_papers)
            logger.info(f"Processing {len(page_papers)} papers (queue depth: {prefetch_queue.qsize()})")

            page_oldest = _oldest_date_in_page(page_papers)
            if page_oldest and (oldest_date is None or page_oldest < oldest_date):
                oldest_date = page_oldest

            result = await _process_page(page_papers, existing_ids, settings, http, pdf_sem, worker_sem)
            stats["new_papers"] += result["new"]
            stats["indexed"] += result["indexed"]
            logger.info(f"Page done: {result['new']} new, {result['indexed']} indexed, {result['skipped']} skipped")

            # 페이지 단위 중간 체크포인트
            _save_checkpoint(settings, {
                "version": CHECKPOINT_VERSION,
                "last_run": datetime.now(timezone.utc).isoformat(),
                "last_collected_date": oldest_date or settings.year_from,
                "year_from": settings.year_from,
                "year_to": settings.year_to,
                "stats": stats,
            })

        await fetch_task
        stats["s2_requests"] = fetcher.request_count
    finally:
        if owns_client:
            await client.aclose()

    # 최종 체크포인트 — window 모드면 year_to를 resume 지점으로 저장
    last_date = oldest_date or settings.year_from
    if settings.ingest_window_days > 0:
        # window 끝 지점을 다음 시작점으로 저장
        last_date = settings.year_to

    _save_checkpoint(settings, {
        "version": CHECKPOINT_VERSION,
        "last_run": datetime.now(timezone.utc).isoformat(),
        "last_collected_date": last_date,
        "year_from": settings.year_from,
        "year_to": settings.year_to,
        "stats": stats,
    })
    logger.info(f"Ingest complete: {stats}")
    return stats


def _run_once() -> None:
    """동기 래퍼 — APScheduler job으로 사용."""
    try:
        asyncio.run(ingest())
    except Exception:
        logger.exception("Ingest failed")


def main() -> None:
    setup_logging()
    settings = get_settings()

    if settings.ingest_cron or settings.ingest_interval > 0:
        from apscheduler.schedulers.blocking import BlockingScheduler
        scheduler = BlockingScheduler()

        if settings.ingest_cron:
            from apscheduler.triggers.cron import CronTrigger
            trigger = CronTrigger.from_crontab(settings.ingest_cron)
            logger.info(f"Scheduled: cron={settings.ingest_cron}")
        else:
            from apscheduler.triggers.interval import IntervalTrigger
            trigger = IntervalTrigger(seconds=settings.ingest_interval)
            logger.info(f"Scheduled: interval={settings.ingest_interval}s")

        # 즉시 1회 실행 후 스케줄 시작
        _run_once()
        scheduler.add_job(_run_once, trigger)
        scheduler.start()
    else:
        asyncio.run(ingest())


if __name__ == "__main__":
    main()
