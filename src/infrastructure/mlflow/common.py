"""Shared helpers for MLflow infrastructure modules."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, TypeVar
import re

import mlflow
from mlflow import MlflowClient
from pydantic import BaseModel

from config import get_mlflow_settings
from infrastructure.mlflow.context import _ACTIVE_EXPERIMENT_NAME
from logging_utils import get_logger

logger = get_logger(__name__)
T = TypeVar("T")


def safe_mlflow_call(description: str, operation: Callable[[], T], default: T) -> T:
    """Run one MLflow operation without letting tracking failures break the app."""

    try:
        return operation()
    except Exception:
        logger.exception("MLflow operation failed: %s", description)
        return default


def ensure_experiment(client: MlflowClient) -> str | None:
    """Create the configured MLflow experiment if needed and return its id."""

    settings = get_mlflow_settings()
    experiment_name = resolve_experiment_name(settings)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id

    experiment_id = settings.create_experiment(client, experiment_name)
    logger.info(
        "Created MLflow experiment name=%s experiment_id=%s settings=%s",
        experiment_name,
        experiment_id,
        type(settings).__name__,
    )
    return experiment_id


def resolve_experiment_name(settings) -> str:
    """Return the active experiment name, honoring any temporary override."""

    return _ACTIVE_EXPERIMENT_NAME.get() or settings.experiment_name


def get_mlflow_client() -> MlflowClient | None:
    """Return a configured MLflow client when tracking is enabled."""

    if not is_tracking_enabled():
        return None

    settings = get_mlflow_settings()
    settings.configure_environment()
    mlflow.set_tracking_uri(settings.tracking_uri)
    return MlflowClient(tracking_uri=settings.tracking_uri)


def is_tracking_enabled() -> bool:
    """Return whether the required MLflow settings object can be used."""

    settings = get_mlflow_settings()
    return settings.is_configured


def to_json_safe(value: Any) -> Any:
    """Recursively convert Pydantic models and containers into JSON-safe data."""

    if isinstance(value, BaseModel):
        return to_json_safe(value.model_dump())
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json_safe(item) for item in value]
    return value


def normalize_name(value: str) -> str:
    """Return a compact lowercase name with non-alphanumeric text replaced by underscores."""

    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def artifact_timestamp() -> str:
    """Return a UTC timestamp formatted for artifact file names."""

    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""

    return datetime.now(timezone.utc).isoformat()
