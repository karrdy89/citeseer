"""로깅 설정 — stream-only."""
from __future__ import annotations

import logging
import sys

from app.core.config import get_settings


def setup_logging() -> None:
    """스트림 전용 로깅 설정. entry point에서 1회 호출."""
    settings = get_settings()

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # httpx/httpcore 로거 레벨 조정 (너무 verbose)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
