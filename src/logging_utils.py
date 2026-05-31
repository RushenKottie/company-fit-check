"""Shared logging helpers for Company Fit Check."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
_LOG_DIR_NAME = ".tmp"
_LOG_FILE_NAME = "company-fit-check.log"
_AZURE_HTTP_LOGGER_NAME = "azure.core.pipeline.policies.http_logging_policy"
_RESPONSE_STATUS_RE = re.compile(r"Response status:\s*'?(\d{3})'?")
_APP_STREAM_HANDLER_ATTR = "_company_fit_check_stream_handler"
_APP_FILE_HANDLER_ATTR = "_company_fit_check_file_handler"
_AZURE_FILTER_ATTR = "_company_fit_check_filter"


class _AzureHttpNoiseFilter(logging.Filter):
    """Reduce Azure SDK HTTP success noise while preserving failed response details."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != _AZURE_HTTP_LOGGER_NAME:
            return True

        message = record.getMessage()
        target_level = self._target_level(message)
        if target_level is None:
            return True

        record.levelno = target_level
        record.levelname = logging.getLevelName(target_level)
        return target_level >= logging.getLogger().getEffectiveLevel()

    def _target_level(self, message: str) -> int | None:
        if message.startswith("Request URL:"):
            return logging.DEBUG

        status_match = _RESPONSE_STATUS_RE.search(message)
        if status_match is None:
            return None

        status_code = int(status_match.group(1))
        if 200 <= status_code < 400:
            return logging.DEBUG
        return logging.WARNING


def _project_root() -> Path:
    """Return the repository root based on this module location."""

    return Path(__file__).resolve().parents[1]


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
    """Configure application logging and restore handlers if a framework resets them."""

    root_logger = logging.getLogger()
    first_configuration = not getattr(configure_logging, "_configured", False)
    file_handler_added = False

    level_name = os.getenv("COMPANY_FIT_CHECK_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT)

    root_logger.setLevel(level)
    if not _has_handler(root_logger, _APP_STREAM_HANDLER_ATTR):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        setattr(stream_handler, _APP_STREAM_HANDLER_ATTR, True)
        root_logger.addHandler(stream_handler)

    if not _has_handler(root_logger, _APP_FILE_HANDLER_ATTR):
        file_handler = logging.FileHandler(
            get_log_file_path(),
            mode="a",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        setattr(file_handler, _APP_FILE_HANDLER_ATTR, True)
        root_logger.addHandler(file_handler)
        file_handler_added = True

    azure_logger = logging.getLogger(_AZURE_HTTP_LOGGER_NAME)
    if not any(getattr(filter_, _AZURE_FILTER_ATTR, False) for filter_ in azure_logger.filters):
        azure_filter = _AzureHttpNoiseFilter()
        setattr(azure_filter, _AZURE_FILTER_ATTR, True)
        azure_logger.addFilter(azure_filter)
    configure_logging._configured = True
    if first_configuration or file_handler_added:
        logging.getLogger(__name__).info(
            "Logging configured file=%s",
            get_log_file_path(),
        )


def _has_handler(logger: logging.Logger, marker_attr: str) -> bool:
    """Return whether a logger already has one of this module's handlers."""

    return any(getattr(handler, marker_attr, False) for handler in logger.handlers)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""

    configure_logging()
    return logging.getLogger(name)
