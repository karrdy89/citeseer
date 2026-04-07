"""es_client 단위 테스트."""
import pytest

from app.es.client import _to_es_doc
from conftest import make_paper


class TestToEsDoc:
    """Paper dict → ES 문서 변환 테스트."""

    def test_basic_fields(self):
        doc = _to_es_doc(make_paper())

        assert doc["s2_id"] == "abc123"
        assert doc["title"] == "Test Paper"
        assert doc["venue"] == "NeurIPS"
        assert doc["content_type"] == "pdf"
        assert "ingested_at" in doc
        assert "updated_at" in doc

    def test_internal_fields_excluded(self):
        doc = _to_es_doc(make_paper())

        assert "_s2_oa_url" not in doc

    def test_embedding_included_when_present(self):
        doc = _to_es_doc(make_paper())
        assert "embedding" in doc
        assert len(doc["embedding"]) == 768

    def test_embedding_excluded_when_none(self):
        doc = _to_es_doc(make_paper(embedding=None))
        assert "embedding" not in doc
