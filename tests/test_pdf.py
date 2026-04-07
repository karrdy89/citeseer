"""pdf URL 단위 테스트."""
import pytest

from app.s2.pdf import get_pdf_url


class TestGetPdfUrl:
    """PDF URL 우선순위 테스트."""

    def test_s2_oa_url_first(self):
        paper = {
            "_s2_oa_url": "https://s2.com/paper.pdf",
            "arxiv_id": "2301.00001",
            "doi": "10.1234/test",
        }
        assert get_pdf_url(paper) == "https://s2.com/paper.pdf"

    def test_arxiv_fallback(self):
        paper = {
            "_s2_oa_url": None,
            "arxiv_id": "2301.00001",
            "doi": None,
        }
        assert get_pdf_url(paper) == "https://arxiv.org/pdf/2301.00001.pdf"

    def test_acl_anthology_fallback(self):
        paper = {
            "_s2_oa_url": None,
            "arxiv_id": None,
            "doi": "10.18653/v1/2023.acl-long.1",  # aclweb 패턴이 아님
        }
        # aclweb이 doi에 없으면 None
        assert get_pdf_url(paper) is None

    def test_acl_anthology_match(self):
        paper = {
            "_s2_oa_url": None,
            "arxiv_id": None,
            "doi": "10.aclweb.org/anthology/2023.acl-long.1",
        }
        url = get_pdf_url(paper)
        assert url is not None
        assert "aclanthology.org" in url
        assert url.endswith(".pdf")

    def test_no_url_available(self):
        paper = {
            "_s2_oa_url": None,
            "arxiv_id": None,
            "doi": None,
        }
        assert get_pdf_url(paper) is None

    def test_s2_oa_takes_priority_over_arxiv(self):
        """S2 OA URL이 있으면 arXiv보다 우선."""
        paper = {
            "_s2_oa_url": "https://openreview.net/pdf?id=xxx",
            "arxiv_id": "2301.00001",
        }
        assert get_pdf_url(paper) == "https://openreview.net/pdf?id=xxx"
