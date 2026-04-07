"""체크포인트 로직 단위 테스트."""
import json

import pytest

from app.core.config import Settings, normalize_date
from app.ingest.pipeline import _apply_checkpoint, _oldest_date_in_page, _save_checkpoint


class TestApplyCheckpoint:
    """체크포인트 resume 로직 테스트."""

    def test_v2_checkpoint_restores_date(self, tmp_path):
        settings = Settings(
            checkpoint_path=str(tmp_path / "cp.json"),
            s2_rate_limit=0,
        )
        checkpoint = {
            "version": 2,
            "last_collected_date": "2024-03-15",
            "year_from": "2020-01-01",
        }
        _apply_checkpoint(checkpoint, settings)
        assert settings.year_from == "2024-03-15"

    def test_legacy_v1_checkpoint_converts_year(self, tmp_path):
        settings = Settings(
            checkpoint_path=str(tmp_path / "cp.json"),
            s2_rate_limit=0,
        )
        checkpoint = {"year_from": 2022}
        _apply_checkpoint(checkpoint, settings)
        assert settings.year_from == "2022-01-01"

    def test_empty_checkpoint_no_change(self, tmp_path):
        settings = Settings(
            checkpoint_path=str(tmp_path / "cp.json"),
            year_from="2021-06-01",
            s2_rate_limit=0,
        )
        _apply_checkpoint({}, settings)
        assert settings.year_from == "2021-06-01"


class TestOldestDateInPage:
    """페이지 내 최소 날짜 추출 테스트."""

    def test_finds_oldest(self):
        papers = [
            {"publication_date": "2024-06-15"},
            {"publication_date": "2024-01-10"},
            {"publication_date": "2024-09-20"},
        ]
        assert _oldest_date_in_page(papers) == "2024-01-10"

    def test_skips_none_dates(self):
        papers = [
            {"publication_date": None},
            {"publication_date": "2024-03-01"},
            {},
        ]
        assert _oldest_date_in_page(papers) == "2024-03-01"

    def test_all_none_returns_none(self):
        papers = [{"publication_date": None}, {}]
        assert _oldest_date_in_page(papers) is None


class TestSaveCheckpoint:
    """체크포인트 저장 테스트."""

    def test_saves_v2_format(self, tmp_path):
        settings = Settings(
            checkpoint_path=str(tmp_path / "data" / "cp.json"),
            s2_rate_limit=0,
        )
        _save_checkpoint(settings, {
            "version": 2,
            "last_run": "2026-04-07T00:00:00+00:00",
            "last_collected_date": "2024-03-15",
            "year_from": "2020-01-01",
            "year_to": "2026-12-31",
            "stats": {"total_fetched": 100},
        })

        with open(settings.checkpoint_path) as f:
            data = json.load(f)
        assert data["version"] == 2
        assert data["last_collected_date"] == "2024-03-15"


class TestNormalizeDate:
    """날짜 정규화 테스트."""

    def test_year_to_start_date(self):
        assert normalize_date("2022", is_start=True) == "2022-01-01"

    def test_year_to_end_date(self):
        assert normalize_date("2025", is_start=False) == "2025-12-31"

    def test_full_date_passthrough(self):
        assert normalize_date("2023-06-15", is_start=True) == "2023-06-15"
        assert normalize_date("2023-06-15", is_start=False) == "2023-06-15"

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            normalize_date("2024/01/01", is_start=True)


class TestSettingsDateResolution:
    """Settings year_from/year_to 날짜 해석 테스트."""

    def test_year_only_resolves_to_date(self):
        s = Settings(year_from="2022", year_to="2025", s2_rate_limit=0)
        assert s.year_from == "2022-01-01"
        assert s.year_to == "2025-12-31"

    def test_full_date_kept(self):
        s = Settings(year_from="2023-06-01", year_to="2024-12-31", s2_rate_limit=0)
        assert s.year_from == "2023-06-01"
        assert s.year_to == "2024-12-31"

    def test_empty_year_to_defaults_to_today(self):
        from datetime import date
        s = Settings(year_from="2020", s2_rate_limit=0)
        assert s.year_to == date.today().isoformat()

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            Settings(year_from="2024/01/01", s2_rate_limit=0)
