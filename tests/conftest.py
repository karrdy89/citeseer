"""공유 fixtures, factories, mock transports."""
from __future__ import annotations

import json

import httpx
import pytest

from app.core.config import Settings


# --- Factories ---

def make_s2_data(**overrides) -> dict:
    """S2 API 응답 단일 논문 데이터."""
    base = {
        "paperId": "abc123",
        "externalIds": {"DOI": "10.1234/test", "ArXiv": "2301.00001"},
        "title": "Test Paper",
        "abstract": "This is abstract.",
        "authors": [{"name": "Alice"}, {"name": "Bob"}],
        "year": 2024,
        "publicationDate": "2024-06-15",
        "venue": "NeurIPS",
        "citationCount": 42,
        "tldr": {"text": "A test paper about testing."},
        "openAccessPdf": {"url": "https://example.com/paper.pdf"},
        "embedding": {"model": "specter_v2", "vector": [0.1] * 768},
    }
    base.update(overrides)
    return base


def make_paper(**overrides) -> dict:
    """내부 Paper dict."""
    base = {
        "s2_id": "abc123",
        "doi": "10.1234/test",
        "arxiv_id": "2301.00001",
        "title": "Test Paper",
        "abstract": "Abstract text.",
        "authors": ["Alice", "Bob"],
        "year": 2024,
        "publication_date": "2024-06-15",
        "venue": "NeurIPS",
        "citation_count": 42,
        "tldr": "Summary.",
        "url": "https://semanticscholar.org/paper/abc123",
        "embedding": [0.1] * 768,
        "content_type": "pdf",
        "pdf_path": "/data/pdfs/NeurIPS/2024/2301.00001.pdf",
        "json_path": None,
        "_s2_oa_url": "https://example.com/paper.pdf",
    }
    base.update(overrides)
    return base


def make_s2_response(
    paper_ids: list[str],
    token: str | None = None,
    publication_date: str = "2024-06-15",
) -> dict:
    """S2 bulk API 응답."""
    return {
        "data": [
            {
                "paperId": pid,
                "externalIds": {"ArXiv": f"2301.{pid}"},
                "title": f"Paper {pid}",
                "abstract": f"Abstract for {pid}.",
                "authors": [{"name": "Author"}],
                "year": 2024,
                "publicationDate": publication_date,
                "venue": "NeurIPS",
                "citationCount": 5,
                "tldr": {"text": f"TLDR {pid}"},
                "openAccessPdf": None,
                "embedding": {"model": "specter_v2", "vector": [0.1] * 768},
            }
            for pid in paper_ids
        ],
        "token": token,
    }


# --- Mock Transports ---

class MockEmbeddingTransport(httpx.AsyncBaseTransport):
    """SPECTER2 /encode mock."""

    def __init__(self):
        self.call_count = 0
        self.last_texts: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        body = json.loads(request.content)
        texts = body["text"]
        self.last_texts = texts
        return httpx.Response(200, json={"embeddings": [[0.1] * 768 for _ in texts]})


class MockS2Transport(httpx.AsyncBaseTransport):
    """S2 API mock."""

    def __init__(self, responses: list[dict | Exception]):
        self._responses = list(responses)
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self._call_count += 1
        if not self._responses:
            return httpx.Response(200, json={"data": [], "token": None})
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return httpx.Response(200, json=resp)


class PipelineMockTransport(httpx.AsyncBaseTransport):
    """S2 + SPECTER2 mock, ES는 실제 프록시."""

    def __init__(self, s2_responses: list[dict]):
        self._s2_responses = list(s2_responses)
        self._real_transport = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "semanticscholar.org" in url:
            if self._s2_responses:
                return httpx.Response(200, json=self._s2_responses.pop(0))
            return httpx.Response(200, json={"data": [], "token": None})
        if ":8000" in url:
            body = json.loads(request.content)
            texts = body.get("text", [])
            return httpx.Response(200, json={"embeddings": [[0.1] * 768 for _ in texts]})
        if ":9200" in url:
            return await self._real_transport.handle_async_request(request)
        return httpx.Response(404)

    async def aclose(self) -> None:
        await self._real_transport.aclose()


# --- Shared Fixtures ---

@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        es_host="http://localhost:9200",
        es_index="papers_test",
        pdf_dir=str(tmp_path / "pdfs"),
        json_dir=str(tmp_path / "json"),
        checkpoint_path=str(tmp_path / "checkpoint.json"),
        s2_rate_limit=0,
    )
