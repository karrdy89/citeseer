"""S2 API 응답 변환 — 순수 함수."""
from __future__ import annotations

VENUE_MAP = {
    "Neural Information Processing Systems": "NeurIPS",
    "International Conference on Learning Representations": "ICLR",
    "International Conference on Machine Learning": "ICML",
    "Annual Meeting of the Association for Computational Linguistics": "ACL",
    "Conference on Empirical Methods in Natural Language Processing": "EMNLP",
    "North American Chapter of the Association for Computational Linguistics": "NAACL",
    "Computer Vision and Pattern Recognition": "CVPR",
    "AAAI Conference on Artificial Intelligence": "AAAI",
    "Conference on Human Factors in Computing Systems": "CHI",
}


def _resolve_venue(data: dict) -> str:
    venue = data.get("venue") or ""
    return VENUE_MAP.get(venue, venue)


def s2_to_paper(data: dict) -> dict:
    """S2 API 응답을 내부 Paper dict로 변환."""
    ext = data.get("externalIds") or {}
    oa = data.get("openAccessPdf")
    emb = data.get("embedding")

    return {
        "doi": ext.get("DOI"),
        "arxiv_id": ext.get("ArXiv"),
        "s2_id": data["paperId"],
        "title": data.get("title", ""),
        "abstract": data.get("abstract") or "",
        "authors": [a["name"] for a in data.get("authors") or []],
        "year": data.get("year"),
        "publication_date": data.get("publicationDate"),
        "venue": _resolve_venue(data),
        "citation_count": data.get("citationCount", 0),
        "tldr": data["tldr"]["text"] if data.get("tldr") else None,
        "url": f"https://semanticscholar.org/paper/{data['paperId']}",
        "embedding": emb["vector"] if emb else None,
        "_s2_oa_url": oa["url"] if oa and oa.get("url") else None,
        "content_type": None,
        "pdf_path": None,
        "json_path": None,
    }
