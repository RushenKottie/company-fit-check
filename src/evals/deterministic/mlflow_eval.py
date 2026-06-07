"""MLflow helpers for deterministic evaluation runs."""

from __future__ import annotations

import json
from typing import Any

import mlflow
from mlflow import MlflowClient

from config import get_mlflow_settings
from evals import EVAL_EXPERIMENT_NAME, repo_root
from evals.deterministic.models import CaseExecutionResult
from evals.deterministic.scorers import ALL_SCORERS


def get_eval_client() -> MlflowClient:
    """Return an MLflow client pointed at the configured tracking URI."""

    settings = get_mlflow_settings()
    mlflow.set_tracking_uri(settings.tracking_uri)
    return MlflowClient(tracking_uri=settings.tracking_uri)


def ensure_eval_experiment() -> str:
    """Create or fetch the deterministic eval experiment."""

    client = get_eval_client()
    experiment = client.get_experiment_by_name(EVAL_EXPERIMENT_NAME)
    if experiment is not None:
        return experiment.experiment_id

    artifact_root = (repo_root() / ".mlflow-eval-artifacts").resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    return client.create_experiment(
        EVAL_EXPERIMENT_NAME,
        artifact_location=artifact_root.as_uri(),
    )


def build_eval_row(result: CaseExecutionResult) -> dict[str, Any]:
    """Build one dataset row for mlflow.genai.evaluate()."""

    return {
        "inputs": result.case_inputs,
        "outputs": result.model_dump(mode="json"),
    }


def evaluate_case_with_mlflow(result: CaseExecutionResult):
    """Log scorer results for one case into the active MLflow run."""

    return mlflow.genai.evaluate(
        data=[build_eval_row(result)],
        scorers=ALL_SCORERS,
    )


def log_json_artifact_to_active_run(payload: dict[str, Any], artifact_path: str) -> None:
    """Log one JSON artifact to the active MLflow run."""

    mlflow.log_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        artifact_file=artifact_path,
    )
