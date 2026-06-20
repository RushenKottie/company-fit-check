"""Manage MLflow datasets used by evaluation runs."""

from __future__ import annotations

from typing import Any
import hashlib
import json

from mlflow import MlflowClient
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag

from infrastructure.mlflow.common import (
    ensure_experiment,
    get_mlflow_client,
    is_tracking_enabled,
    to_json_safe,
    normalize_name,
    safe_mlflow_call,
    utc_now_iso,
)
from logging_utils import get_logger

logger = get_logger(__name__)
_CASE_PAYLOAD_HASH_IGNORED_FIELDS = frozenset(
    {
        "pdf_path",
        "case_path",
        "run_id",
        "suite_stamp",
        "created_at_utc",
        "updated_at_utc",
        "generated_at_utc",
    }
)


def ensure_case_dataset_for_run(
    run_id: str,
    *,
    case_id: int,
    case_name: str,
    case_payload: dict[str, Any],
    case_source: str = "regression",
) -> str | None:
    """Create or reuse one MLflow dataset for an evaluation case and link the run to it."""

    client, experiment_id = _get_dataset_tracking_context(run_id)
    if client is None or experiment_id is None:
        return None

    payload_hash = _case_payload_hash(case_payload)
    dataset_name = _build_case_dataset_name(
        case_id=case_id,
        case_name=case_name,
        case_payload=case_payload,
        case_source=case_source,
    )
    dataset = _ensure_case_dataset(
        client,
        experiment_id=experiment_id,
        dataset_name=dataset_name,
        case_id=case_id,
        case_name=case_name,
        case_payload=case_payload,
        case_source=case_source,
    )
    if dataset is None:
        return None

    _tag_run_with_case(
        client,
        run_id,
        case_id=case_id,
        case_name=case_name,
        case_source=case_source,
        payload_hash=payload_hash,
    )
    _link_run_to_case_dataset(
        client,
        run_id,
        dataset,
        case_id=case_id,
        case_source=case_source,
        payload_hash=payload_hash,
    )

    logger.info(
        "Linked MLflow run to case dataset run_id=%s dataset_id=%s dataset_name=%s",
        run_id,
        dataset.dataset_id,
        dataset_name,
    )
    return dataset.dataset_id


def _get_dataset_tracking_context(run_id: str) -> tuple[MlflowClient | None, str | None]:
    """Return the MLflow client and experiment for dataset logging."""

    if not run_id or not is_tracking_enabled():
        return None, None

    client = get_mlflow_client()
    if client is None:
        return None, None

    experiment_id = safe_mlflow_call(
        "ensure MLflow experiment",
        lambda: ensure_experiment(client),
        None,
    )
    return client, experiment_id


def _ensure_case_dataset(
    client: MlflowClient,
    *,
    experiment_id: str,
    dataset_name: str,
    case_id: int,
    case_name: str,
    case_payload: dict[str, Any],
    case_source: str,
):
    """Return the MLflow dataset for one evaluation case."""

    dataset_result = safe_mlflow_call(
        "get or create case dataset",
        lambda: _get_or_create_case_dataset(
            client,
            experiment_id=experiment_id,
            dataset_name=dataset_name,
            case_id=case_id,
            case_name=case_name,
            case_source=case_source,
        ),
        None,
    )
    if dataset_result is None:
        return None

    dataset, created = dataset_result
    if created:
        safe_mlflow_call(
            "set case dataset tags",
            lambda: client.set_dataset_tags(
                dataset.dataset_id,
                _build_case_dataset_tags(
                    case_id=case_id,
                    case_name=case_name,
                    case_payload=case_payload,
                    case_source=case_source,
                ),
            ),
            None,
        )
        dataset = safe_mlflow_call(
            "upsert case dataset record",
            lambda: _upsert_case_dataset_record(
                client,
                dataset,
                case_name=case_name,
                case_payload=case_payload,
            ),
            dataset,
        )

    return dataset


def _tag_run_with_case(
    client: MlflowClient,
    run_id: str,
    *,
    case_id: int,
    case_name: str,
    case_source: str,
    payload_hash: str,
) -> None:
    """Add evaluation case details to the MLflow run."""

    tags = {
        "case_source": case_source,
        "case_id": str(case_id),
        "case_name": case_name,
        "case_payload_hash": payload_hash,
    }
    for key, value in tags.items():
        safe_mlflow_call(
            f"set {key} tag",
            lambda key=key, value=value: client.set_tag(run_id, key, value),
            None,
        )


def _link_run_to_case_dataset(
    client: MlflowClient,
    run_id: str,
    dataset,
    *,
    case_id: int,
    case_source: str,
    payload_hash: str,
) -> None:
    """Link the MLflow run to the evaluation case dataset."""

    safe_mlflow_call(
        "link MLflow run to case dataset",
        lambda: client.log_inputs(
            run_id,
            datasets=[
                DatasetInput(
                    dataset=dataset._to_mlflow_entity(),
                    tags=[
                        InputTag(key="mlflow.data.context", value="evaluation"),
                        InputTag(key="case_source", value=case_source),
                        InputTag(key="case_id", value=str(case_id)),
                        InputTag(key="case_payload_hash", value=payload_hash),
                    ],
                )
            ],
        ),
        None,
    )


def _build_case_dataset_name(
    *,
    case_id: int,
    case_name: str,
    case_payload: dict[str, Any],
    case_source: str,
) -> str:
    """Return one stable MLflow dataset name for an evaluation case."""

    source = normalize_name(case_source) or "regression"
    payload = to_json_safe(case_payload)
    profession = normalize_name(str(payload.get("profession") or case_name))
    payload_hash = _case_payload_hash(payload)
    if profession:
        return f"{source}_case_{case_id}_{profession}_{payload_hash}"
    return f"{source}_case_{case_id}_{payload_hash}"


def _get_or_create_case_dataset(
    client: MlflowClient,
    *,
    experiment_id: str,
    dataset_name: str,
    case_id: int,
    case_name: str,
    case_source: str,
):
    """Find the case dataset or create it if needed."""

    matches = client.search_datasets(
        filter_string=f"name = '{dataset_name}'",
        max_results=2,
    )
    if matches:
        dataset = matches[0]
        if experiment_id not in dataset.experiment_ids:
            dataset = client.add_dataset_to_experiments(
                dataset.dataset_id,
                [experiment_id],
            )
        return dataset, False

    return (
        client.create_dataset(
            name=dataset_name,
            experiment_id=[experiment_id],
            tags={
                "suite": f"llm_{case_source}",
                "case_source": case_source,
                "case_id": str(case_id),
                "case_name": case_name,
            },
        ),
        True,
    )


def _build_case_dataset_tags(
    *,
    case_id: int,
    case_name: str,
    case_payload: dict[str, Any],
    case_source: str,
) -> dict[str, Any]:
    """Return dataset tags that keep one readable case summary on the dataset object."""

    payload = to_json_safe(case_payload)
    tags = {
        "suite": f"llm_{case_source}",
        "case_source": case_source,
        "case_id": str(case_id),
        "case_name": case_name,
        "profession": str(payload.get("profession") or ""),
        "pdf_path": str(payload.get("pdf_path") or ""),
        "updated_at_utc": utc_now_iso(),
    }
    tags["case_payload_hash"] = _case_payload_hash(payload)
    return tags


def _upsert_case_dataset_record(
    client: MlflowClient,
    dataset,
    *,
    case_name: str,
    case_payload: dict[str, Any],
):
    """Save the case payload as a record in the MLflow dataset."""

    payload = to_json_safe(case_payload)
    _upsert_dataset_records(
        dataset.dataset_id,
        [_build_case_dataset_record(case_name=case_name, case_payload=payload)],
    )
    return client.get_dataset(dataset.dataset_id)


def _build_case_dataset_record(
    *,
    case_name: str,
    case_payload: dict[str, Any],
) -> dict[str, Any]:
    """Return one MLflow dataset record preserving the case JSON in inputs."""

    payload = to_json_safe(case_payload)
    return {
        "inputs": payload,
        "tags": {
            "case_name": case_name,
            "profession": str(payload.get("profession") or ""),
            "case_payload_hash": _case_payload_hash(payload),
            "updated_at_utc": utc_now_iso(),
        },
    }


def _upsert_dataset_records(dataset_id: str, records: list[dict[str, Any]]) -> None:
    """Write records without MLflow's public session-field validation."""

    from mlflow.tracking._tracking_service.utils import _get_store

    # Public merge_records() reserves top-level "goal" for session datasets, but
    # case inputs must match test-case.json exactly.
    _get_store().upsert_dataset_records(dataset_id=dataset_id, records=records)


def _case_payload_hash(payload: Any) -> str:
    """Return a stable short hash for one case payload."""

    normalized = json.dumps(
        _case_identity_payload(payload),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _case_identity_payload(payload: Any) -> Any:
    """Return the case fields that define dataset identity."""

    value = to_json_safe(payload)
    if not isinstance(value, dict):
        return value
    return {
        key: item
        for key, item in value.items()
        if key not in _CASE_PAYLOAD_HASH_IGNORED_FIELDS
    }
