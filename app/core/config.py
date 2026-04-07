"""Settings — pydantic-settings 기반 설정."""
from __future__ import annotations

import re
from datetime import date
from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_YEAR_RE = re.compile(r"^\d{4}$")


def normalize_date(value: str, is_start: bool) -> str:
    """연도('2020') 또는 날짜('2020-01-15') → YYYY-MM-DD 정규화."""
    value = value.strip()
    if _DATE_RE.match(value):
        return value
    if _YEAR_RE.match(value):
        return f"{value}-01-01" if is_start else f"{value}-12-31"
    raise ValueError(f"날짜는 YYYY 또는 YYYY-MM-DD 형식이어야 합니다: '{value}'")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # S2 API
    s2_api_key: str = ""
    s2_base_url: str = "https://api.semanticscholar.org/graph/v1"
    s2_fields: str = (
        "paperId,externalIds,title,abstract,authors,"
        "year,venue,publicationVenue,citationCount,"
        "openAccessPdf,publicationDate"
    )

    # Target venues
    target_venues: list[str] = [
        "NeurIPS", "ICLR", "ICML",
        "ACL", "EMNLP", "NAACL",
        "CVPR", "AAAI", "CHI",
    ]

    # 수집 범위 — YYYY 또는 YYYY-MM-DD (env: YEAR_FROM, YEAR_TO)
    # year_to 미설정 시 오늘 날짜 → 항상 최신 논문까지 수집
    year_from: str = "2020"
    year_to: str = ""

    @model_validator(mode="after")
    def _normalize_dates(self) -> "Settings":
        self.year_from = normalize_date(self.year_from, is_start=True)
        if not self.year_to:
            self.year_to = date.today().isoformat()
        else:
            self.year_to = normalize_date(self.year_to, is_start=False)
        return self

    # Elasticsearch
    es_host: str = "http://localhost:9200"
    es_index: str = "papers"

    # SPECTER2 서빙
    specter_url: str = "http://localhost:8000"

    # 데이터 경로
    data_dir: str = "./data"
    pdf_dir: str = "./data/pdfs"
    json_dir: str = "./data/json"
    checkpoint_path: str = "./data/checkpoint.json"

    # 파이프라인 제어
    s2_max_requests: int = 0
    s2_rate_limit: float = 1.0
    max_retries: int = 3
    prefetch_buffer: int = 2
    sub_batch_size: int = 100
    worker_concurrency: int = 4
    pdf_concurrency: int = 10
    embedding_batch_size: int = 32

    # 타임아웃 (초)
    s2_timeout: float = 30.0
    specter_timeout: float = 120.0
    pdf_timeout: float = 60.0
    es_timeout: float = 60.0

    # 배치 윈도우 (일) — 1회 실행 시 수집할 날짜 범위 폭 (0=전체)
    ingest_window_days: int = 0

    # 스케줄링 — cron 표현식 또는 interval (초). cron 우선.
    ingest_cron: str = ""
    ingest_interval: int = 0

    # 로깅
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
