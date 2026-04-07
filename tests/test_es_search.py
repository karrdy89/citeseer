"""search 모듈 테스트."""
import json

import httpx
import pytest

from app.core.config import Settings
from app.es.search import build_query, search_papers, _format_result
from app.es.client import ensure_index, bulk_upsert


class TestBuildQuery:
    """build_query 단위 테스트."""

    def test_basic_query(self):
        q = build_query("attention", limit=5)
        assert q["size"] == 5
        should = q["query"]["bool"]["should"]
        assert should[0]["multi_match"]["query"] == "attention"
        assert should[0]["multi_match"]["boost"] == 0.3

    def test_venue_filter(self):
        q = build_query("test", venues=["NeurIPS", "ICLR"])
        filters = q["query"]["bool"]["filter"]
        venue_filter = [f for f in filters if "terms" in f]
        assert len(venue_filter) == 1
        assert venue_filter[0]["terms"]["venue"] == ["NeurIPS", "ICLR"]

    def test_no_venue_filter(self):
        q = build_query("test", venues=None)
        filters = q["query"]["bool"]["filter"]
        venue_filter = [f for f in filters if "terms" in f]
        assert len(venue_filter) == 0

    def test_date_defaults(self):
        from datetime import datetime
        now_year = datetime.now().year
        q = build_query("test")
        filters = q["query"]["bool"]["filter"]
        date_filter = [f for f in filters if "range" in f and "publication_date" in f["range"]]
        assert len(date_filter) == 1
        assert date_filter[0]["range"]["publication_date"]["gte"] == f"{now_year - 1}-01-01"
        assert date_filter[0]["range"]["publication_date"]["lte"] == f"{now_year}-12-31"

    def test_explicit_dates(self):
        q = build_query("test", date_from="2020-03-01", date_to="2023-12-31")
        filters = q["query"]["bool"]["filter"]
        date_filter = [f for f in filters if "range" in f and "publication_date" in f["range"]]
        assert date_filter[0]["range"]["publication_date"]["gte"] == "2020-03-01"
        assert date_filter[0]["range"]["publication_date"]["lte"] == "2023-12-31"

    def test_with_query_vector(self):
        vec = [0.1] * 768
        q = build_query("test", query_vector=vec, limit=10)
        assert "knn" in q
        assert q["knn"]["field"] == "embedding"
        assert q["knn"]["query_vector"] == vec
        assert q["knn"]["boost"] == 0.7
        assert q["knn"]["k"] == 20

    def test_without_query_vector(self):
        q = build_query("test")
        assert "knn" not in q

    def test_excludes_embedding(self):
        q = build_query("test")
        assert "embedding" in q["_source"]["excludes"]


class TestFormatResult:
    """_format_result 단위 테스트."""

    def test_basic(self):
        hit = {
            "_source": {
                "title": "Test Paper",
                "abstract": "Abstract.",
                "authors": ["Alice"],
                "year": 2024,
                "publication_date": "2024-06-15",
                "venue": "NeurIPS",
                "url": "https://semanticscholar.org/paper/abc",
                "arxiv_id": "2301.00001",
            }
        }
        result = _format_result(hit)
        assert result["title"] == "Test Paper"
        assert result["publication_date"] == "2024-06-15"
        assert result["url"] == "https://arxiv.org/pdf/2301.00001.pdf"

    def test_no_arxiv_uses_s2_url(self):
        hit = {
            "_source": {
                "title": "No arXiv",
                "abstract": "",
                "authors": [],
                "year": 2024,
                "venue": "CHI",
                "url": "https://semanticscholar.org/paper/xyz",
                "arxiv_id": None,
            }
        }
        result = _format_result(hit)
        assert result["url"] == "https://semanticscholar.org/paper/xyz"


# --- 통합 테스트 (ES 필요) ---

pytestmark_integration = pytest.mark.integration


class MockSearchTransport(httpx.AsyncBaseTransport):
    """SPECTER2 mock + ES 실제 프록시."""

    def __init__(self):
        self._real_transport = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)

        # SPECTER2 → mock
        if ":8000" in url:
            body = json.loads(request.content)
            texts = body.get("text", [])
            embeddings = [[0.1] * 768 for _ in texts]
            return httpx.Response(200, json={"embeddings": embeddings})

        # ES → 실제
        return await self._real_transport.handle_async_request(request)

    async def aclose(self) -> None:
        await self._real_transport.aclose()


@pytest.fixture
def search_settings(tmp_path):
    return Settings(
        es_host="http://localhost:9200",
        es_index="papers_search_test",
    )


@pytest.fixture
async def populated_index(search_settings):
    """테스트 데이터가 있는 ES 인덱스."""
    async with httpx.AsyncClient() as client:
        await client.delete(f"{search_settings.es_host}/{search_settings.es_index}")
        await ensure_index(search_settings, client)

        papers = [
            {
                "s2_id": f"search_test_{i}",
                "doi": None,
                "arxiv_id": f"2301.{i:05d}" if i % 2 == 0 else None,
                "title": f"Attention Mechanism in Transformers Part {i}",
                "abstract": f"We study attention mechanisms for NLP tasks. Paper {i}.",
                "authors": ["Author A", "Author B"],
                "year": 2024 if i < 3 else 2023,
                "publication_date": "2024-06-15" if i < 3 else "2023-03-01",
                "venue": "NeurIPS" if i < 3 else "ICLR",
                "citation_count": i * 10,
                "tldr": f"Summary {i}",
                "url": f"https://semanticscholar.org/paper/search_test_{i}",
                "embedding": [0.1 + i * 0.01] * 768,
                "content_type": "abstract_only",
                "pdf_path": None,
                "json_path": None,
            }
            for i in range(5)
        ]
        await bulk_upsert(papers, search_settings, client)
        await client.post(f"{search_settings.es_host}/{search_settings.es_index}/_refresh")

        yield

        await client.delete(f"{search_settings.es_host}/{search_settings.es_index}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_returns_results(search_settings, populated_index):
    transport = MockSearchTransport()
    async with httpx.AsyncClient(transport=transport) as client:
        results = await search_papers(
            "attention transformers",
            settings=search_settings,
            client=client,
            date_from="2023-01-01",
            date_to="2024-12-31",
            limit=5,
        )
    assert len(results) > 0
    for r in results:
        assert "title" in r
        assert "abstract" in r
        assert "authors" in r
        assert "year" in r
        assert "venue" in r
        assert "url" in r


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_venue_filter(search_settings, populated_index):
    transport = MockSearchTransport()
    async with httpx.AsyncClient(transport=transport) as client:
        results = await search_papers(
            "attention",
            settings=search_settings,
            client=client,
            venues=["ICLR"],
            date_from="2023-01-01",
            date_to="2024-12-31",
        )
    for r in results:
        assert r["venue"] == "ICLR"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_arxiv_url(search_settings, populated_index):
    """arxiv_id가 있는 논문은 arxiv PDF URL을 반환."""
    transport = MockSearchTransport()
    async with httpx.AsyncClient(transport=transport) as client:
        results = await search_papers(
            "attention",
            settings=search_settings,
            client=client,
            date_from="2023-01-01",
            date_to="2024-12-31",
            limit=10,
        )
    arxiv_results = [r for r in results if "arxiv.org/pdf" in r["url"]]
    assert len(arxiv_results) > 0
