"""MCP 서버 — search_papers 도구."""
from __future__ import annotations

import httpx
from mcp.server.fastmcp import FastMCP

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.es.search import search_papers

setup_logging()

server = FastMCP("citeseer")
settings = get_settings()


@server.tool()
async def search_papers_tool(
    query: str,
    venues: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """탑 학회 논문 하이브리드 검색 (키워드 + 시맨틱).

    venues로 학회 필터링 가능:
      NeurIPS, ICLR, ICML, ACL, EMNLP, NAACL, CVPR, AAAI, CHI
    venues 없으면 전체 학회 검색.

    Args:
        query: 검색 쿼리
        venues: 학회 필터. 기본값 전체.
        date_from: 시작 날짜 (YYYY-MM-DD 또는 YYYY). 기본값 작년.
        date_to: 끝 날짜 (YYYY-MM-DD 또는 YYYY). 기본값 올해.
        limit: 결과 수. 기본값 10.

    Returns:
        논문 리스트. 각 논문: title, abstract, authors, year, publication_date, venue, url
    """
    async with httpx.AsyncClient() as client:
        return await search_papers(
            query=query, settings=settings, client=client,
            venues=venues, date_from=date_from, date_to=date_to, limit=limit,
        )


if __name__ == "__main__":
    server.run()
