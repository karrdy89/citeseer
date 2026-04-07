"""specter client 단위 테스트."""
import pytest

import httpx

from app.core.config import Settings
from app.specter.client import embed_papers, embed_batch
from conftest import MockEmbeddingTransport


@pytest.mark.asyncio
async def test_embed_batch_basic():
    transport = MockEmbeddingTransport()
    settings = Settings()
    async with httpx.AsyncClient(transport=transport) as client:
        vecs = await embed_batch(["hello", "world"], settings, client)
    assert len(vecs) == 2
    assert len(vecs[0]) == 768


@pytest.mark.asyncio
async def test_embed_batch_empty():
    transport = MockEmbeddingTransport()
    settings = Settings()
    async with httpx.AsyncClient(transport=transport) as client:
        vecs = await embed_batch([], settings, client)
    assert vecs == []
    assert transport.call_count == 0


@pytest.mark.asyncio
async def test_embed_papers_skips_existing():
    """S2에서 이미 임베딩을 받은 논문은 스킵."""
    transport = MockEmbeddingTransport()
    settings = Settings(embedding_batch_size=10)

    papers = [
        {"s2_id": "a", "title": "A", "abstract": "aa", "embedding": [0.5] * 768},
        {"s2_id": "b", "title": "B", "abstract": "bb", "embedding": [0.5] * 768},
    ]

    async with httpx.AsyncClient(transport=transport) as client:
        await embed_papers(papers, settings, client)

    # 이미 있으므로 호출 안 함
    assert transport.call_count == 0
    # 기존 임베딩 유지
    assert papers[0]["embedding"] == [0.5] * 768


@pytest.mark.asyncio
async def test_embed_papers_fills_missing():
    """임베딩 없는 논문만 SPECTER2로 요청."""
    transport = MockEmbeddingTransport()
    settings = Settings(embedding_batch_size=10)

    papers = [
        {"s2_id": "a", "title": "A", "abstract": "aa", "embedding": [0.5] * 768},
        {"s2_id": "b", "title": "B", "abstract": "bb", "embedding": None},
        {"s2_id": "c", "title": "C", "abstract": "cc", "embedding": None},
    ]

    async with httpx.AsyncClient(transport=transport) as client:
        await embed_papers(papers, settings, client)

    assert transport.call_count == 1  # 1 batch for 2 papers
    assert len(transport.last_texts) == 2
    # b와 c에 임베딩이 채워짐
    assert papers[1]["embedding"] is not None
    assert papers[2]["embedding"] is not None
    assert len(papers[1]["embedding"]) == 768
    # a는 원본 유지
    assert papers[0]["embedding"] == [0.5] * 768


@pytest.mark.asyncio
async def test_embed_papers_batch_split():
    """embedding_batch_size보다 많으면 여러 번 호출."""
    transport = MockEmbeddingTransport()
    settings = Settings(embedding_batch_size=2)

    papers = [
        {"s2_id": f"p{i}", "title": f"T{i}", "abstract": f"A{i}", "embedding": None}
        for i in range(5)
    ]

    async with httpx.AsyncClient(transport=transport) as client:
        await embed_papers(papers, settings, client)

    # 5개를 batch_size=2로 → 3번 호출 (2+2+1)
    assert transport.call_count == 3
    for p in papers:
        assert p["embedding"] is not None
