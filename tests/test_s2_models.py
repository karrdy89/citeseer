"""s2_to_paper 변환 단위 테스트."""
import pytest

from app.s2.models import s2_to_paper
from conftest import make_s2_data


class TestS2ToPaper:
    """S2 API 응답 → 내부 Paper dict 변환 테스트."""

    def test_basic_conversion(self):
        data = make_s2_data()
        paper = s2_to_paper(data)

        assert paper["s2_id"] == "abc123"
        assert paper["doi"] == "10.1234/test"
        assert paper["arxiv_id"] == "2301.00001"
        assert paper["title"] == "Test Paper"
        assert paper["authors"] == ["Alice", "Bob"]
        assert paper["year"] == 2024
        assert paper["publication_date"] == "2024-06-15"
        assert paper["venue"] == "NeurIPS"
        assert paper["citation_count"] == 42
        assert paper["tldr"] == "A test paper about testing."
        assert paper["_s2_oa_url"] == "https://example.com/paper.pdf"
        assert len(paper["embedding"]) == 768

    def test_missing_optional_fields(self):
        data = make_s2_data(
            externalIds={},
            abstract=None,
            authors=None,
            tldr=None,
            openAccessPdf=None,
            embedding=None,
        )
        paper = s2_to_paper(data)

        assert paper["doi"] is None
        assert paper["arxiv_id"] is None
        assert paper["abstract"] == ""
        assert paper["authors"] == []
        assert paper["tldr"] is None
        assert paper["_s2_oa_url"] is None
        assert paper["embedding"] is None

    def test_content_fields_initially_none(self):
        paper = s2_to_paper(make_s2_data())

        assert paper["content_type"] is None
        assert paper["pdf_path"] is None
        assert paper["json_path"] is None

    def test_url_format(self):
        paper = s2_to_paper(make_s2_data(paperId="xyz789"))
        assert paper["url"] == "https://semanticscholar.org/paper/xyz789"
