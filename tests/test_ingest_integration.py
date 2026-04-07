"""ingest 통합 테스트 — 실제 ES + mock S2 API."""
import asyncio
import json

import httpx
import pytest

from app.core.config import Settings
from app.es.client import ensure_index, get_existing_ids, bulk_upsert
from app.s2.models import s2_to_paper
from conftest import make_s2_data


# ES가 떠있어야 실행되는 테스트
pytestmark = pytest.mark.integration


@pytest.fixture
def settings(tmp_path):
    return Settings(
        es_host="http://localhost:9200",
        es_index="papers_test",
        pdf_dir=str(tmp_path / "pdfs"),
        json_dir=str(tmp_path / "json"),
        checkpoint_path=str(tmp_path / "checkpoint.json"),
    )


@pytest.fixture
async def setup_index(settings):
    """테스트용 인덱스 생성 → 테스트 후 삭제."""
    async with httpx.AsyncClient() as client:
        # 기존 테스트 인덱스 삭제
        await client.delete(f"{settings.es_host}/{settings.es_index}")
        await ensure_index(settings, client)
        yield
        await client.delete(f"{settings.es_host}/{settings.es_index}")


@pytest.mark.asyncio
async def test_ensure_index_creates_index(settings, setup_index):
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{settings.es_host}/{settings.es_index}")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_bulk_upsert_and_get_existing_ids(settings, setup_index):
    papers = [s2_to_paper(make_s2_data(paperId=f"paper{i}")) for i in range(3)]
    for p in papers:
        p["content_type"] = "abstract_only"

    async with httpx.AsyncClient() as client:
        indexed = await bulk_upsert(papers, settings, client)
        assert indexed == 3

        # ES refresh
        await client.post(f"{settings.es_host}/{settings.es_index}/_refresh")

        existing = await get_existing_ids(settings, client)
        assert existing == {"paper0", "paper1", "paper2"}


@pytest.mark.asyncio
async def test_duplicate_skip(settings, setup_index):
    paper = s2_to_paper(make_s2_data(paperId="dup001"))
    paper["content_type"] = "abstract_only"

    async with httpx.AsyncClient() as client:
        await bulk_upsert([paper], settings, client)
        await client.post(f"{settings.es_host}/{settings.es_index}/_refresh")

        existing = await get_existing_ids(settings, client)
        assert "dup001" in existing

        # 같은 ID로 다시 upsert → 에러 없이 덮어쓰기
        indexed = await bulk_upsert([paper], settings, client)
        assert indexed == 1
