"""ES 하이브리드 검색 — BM25 + knn."""
from __future__ import annotations

import logging
from datetime import datetime

import httpx

from app.core.config import Settings
from app.specter.client import embed_batch

logger = logging.getLogger(__name__)


def build_query(
    query: str,
    venues: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 10,
    query_vector: list[float] | None = None,
) -> dict:
    now_year = datetime.now().year
    if date_from is None:
        date_from = f"{now_year - 1}-01-01"
    if date_to is None:
        date_to = f"{now_year}-12-31"

    filters: list[dict] = []
    if venues:
        filters.append({"terms": {"venue": venues}})
    filters.append({"range": {"publication_date": {"gte": date_from, "lte": date_to}}})

    es_query: dict = {
        "size": limit,
        "query": {
            "bool": {
                "should": [{"multi_match": {"query": query, "fields": ["title^3", "abstract", "tldr"], "boost": 0.3}}],
                "filter": filters,
            }
        },
        "_source": {
            "includes": ["title", "abstract", "authors", "year", "publication_date", "venue", "url", "pdf_path", "arxiv_id", "doi"],
            "excludes": ["embedding"],
        },
    }

    if query_vector:
        knn: dict = {
            "field": "embedding", "query_vector": query_vector,
            "k": limit * 2, "num_candidates": 200, "boost": 0.7,
        }
        if filters:
            knn["filter"] = {"bool": {"filter": filters}}
        es_query["knn"] = knn

    return es_query


def _format_result(hit: dict) -> dict:
    src = hit.get("_source", {})
    url = src.get("url", "")
    if src.get("arxiv_id"):
        url = f"https://arxiv.org/pdf/{src['arxiv_id']}.pdf"
    return {
        "title": src.get("title", ""),
        "abstract": src.get("abstract", ""),
        "authors": src.get("authors", []),
        "year": src.get("year"),
        "publication_date": src.get("publication_date"),
        "venue": src.get("venue", ""),
        "url": url,
    }


async def search_papers(
    query: str, settings: Settings, client: httpx.AsyncClient,
    venues: list[str] | None = None, date_from: str | None = None,
    date_to: str | None = None, limit: int = 10,
) -> list[dict]:
    vectors = await embed_batch([query], settings, client)
    query_vector = vectors[0] if vectors else None
    es_query = build_query(query=query, venues=venues, date_from=date_from, date_to=date_to, limit=limit, query_vector=query_vector)
    resp = await client.post(f"{settings.es_host}/{settings.es_index}/_search", json=es_query, timeout=settings.es_timeout)
    resp.raise_for_status()
    hits = resp.json().get("hits", {}).get("hits", [])
    return [_format_result(h) for h in hits]
