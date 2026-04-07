"""파이프라인 통합 테스트 — mock S2 + 실제 ES."""
import httpx
import pytest

from app.core.config import Settings
from app.ingest.pipeline import ingest
from conftest import make_s2_response, PipelineMockTransport


pytestmark = pytest.mark.integration


@pytest.fixture
def pipeline_settings(tmp_path):
    return Settings(
        es_host="http://localhost:9200",
        es_index="papers_pipeline_test",
        pdf_dir=str(tmp_path / "pdfs"),
        json_dir=str(tmp_path / "json"),
        checkpoint_path=str(tmp_path / "checkpoint.json"),
        s2_rate_limit=0,
        s2_max_requests=5,
        sub_batch_size=50,
        worker_concurrency=2,
        pdf_concurrency=3,
        embedding_batch_size=10,
        prefetch_buffer=2,
    )


@pytest.fixture
async def clean_test_index(pipeline_settings):
    """테스트 인덱스 생성 → 테스트 후 삭제."""
    async with httpx.AsyncClient() as client:
        await client.delete(f"{pipeline_settings.es_host}/{pipeline_settings.es_index}")
        yield
        await client.delete(f"{pipeline_settings.es_host}/{pipeline_settings.es_index}")


@pytest.mark.asyncio
async def test_full_pipeline(pipeline_settings, clean_test_index):
    """mock S2 2페이지 → 전체 파이프라인 → ES에 적재 확인."""
    s2_responses = [
        make_s2_response(["p1", "p2", "p3"], token="next"),
        make_s2_response(["p4", "p5"], token=None),
    ]
    transport = PipelineMockTransport(s2_responses)

    async with httpx.AsyncClient(transport=transport) as client:
        stats = await ingest(pipeline_settings, client=client)

    assert stats["total_fetched"] == 5
    assert stats["new_papers"] == 5
    assert stats["indexed"] == 5
    assert stats["s2_requests"] == 2

    # ES에 실제로 들어갔는지 확인
    async with httpx.AsyncClient() as client:
        await client.post(f"{pipeline_settings.es_host}/{pipeline_settings.es_index}/_refresh")
        resp = await client.get(
            f"{pipeline_settings.es_host}/{pipeline_settings.es_index}/_count"
        )
        assert resp.json()["count"] == 5


@pytest.mark.asyncio
async def test_incremental_dedup(pipeline_settings, clean_test_index):
    """같은 데이터로 두 번 실행 → 두 번째는 중복 스킵."""
    s2_responses = [
        make_s2_response(["p1", "p2"], token=None),
    ]

    # 첫 번째 실행
    transport1 = PipelineMockTransport(list(s2_responses))
    async with httpx.AsyncClient(transport=transport1) as client:
        stats1 = await ingest(pipeline_settings, client=client)
    assert stats1["new_papers"] == 2
    assert stats1["indexed"] == 2

    # ES refresh
    async with httpx.AsyncClient() as client:
        await client.post(f"{pipeline_settings.es_host}/{pipeline_settings.es_index}/_refresh")

    # 두 번째 실행 → 중복이므로 new=0
    transport2 = PipelineMockTransport(list(s2_responses))
    async with httpx.AsyncClient(transport=transport2) as client:
        stats2 = await ingest(pipeline_settings, client=client)
    assert stats2["total_fetched"] == 2
    assert stats2["new_papers"] == 0
    assert stats2["indexed"] == 0
