"""Shared logging helpers for Company Fit Check."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
_LOG_DIR_NAME = ".tmp"
_LOG_FILE_NAME = "company-fit-check.log"


def _project_root() -> Path:
    """Return the repository root based on this module location."""

    return Path(__file__).resolve().parents[2]


def get_log_file_path() -> Path:
    """Return the on-disk log file path used by the app."""

    return get_tmp_dir_path() / _LOG_FILE_NAME


def get_tmp_dir_path() -> Path:
    """Return the project-local temporary directory path."""

    log_dir = _project_root() / _LOG_DIR_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def write_debug_text_artifact(prefix: str, content: str) -> Path:
    """Persist a timestamped debug text artifact under the project tmp directory."""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    artifact_path = get_tmp_dir_path() / f"{prefix}-{timestamp}.txt"
    artifact_path.write_text(content, encoding="utf-8")
    logging.getLogger(__name__).info(
        "Wrote debug artifact path=%s chars=%s",
        artifact_path,
        len(content),
    )
    return artifact_path


def configure_logging() -> None:
    """Configure application logging once per process."""

    root_logger = logging.getLogger()
    if getattr(configure_logging, "_configured", False):
        return

    level_name = os.getenv("COMPANY_FIT_CHECK_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(get_log_file_path(), encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    configure_logging._configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""

    configure_logging()
    return logging.getLogger(name)
