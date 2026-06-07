"""MLflow dataset linking used by evaluation runs."""

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
    json_ready,
    safe_mlflow_call,
    slugify,
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

    if not run_id or not is_tracking_enabled():
        return None

    client = get_mlflow_client()
    if client is None:
        return None

    experiment_id = safe_mlflow_call(
        "ensure MLflow experiment",
        lambda: ensure_experiment(client),
        None,
    )
    if experiment_id is None:
        return None

    payload_hash = _case_payload_hash(case_payload)
    dataset_name = _build_case_dataset_name(
        case_id=case_id,
        case_name=case_name,
        case_payload=case_payload,
        case_source=case_source,
    )
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
    safe_mlflow_call(
        "set case source tag",
        lambda: client.set_tag(run_id, "case_source", case_source),
        None,
    )
    safe_mlflow_call(
        "set case id tag",
        lambda: client.set_tag(run_id, "case_id", str(case_id)),
        None,
    )
    safe_mlflow_call(
        "set case name tag",
        lambda: client.set_tag(run_id, "case_name", case_name),
        None,
    )
    safe_mlflow_call(
        "set case payload hash tag",
        lambda: client.set_tag(run_id, "case_payload_hash", payload_hash),
        None,
    )
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

    logger.info(
        "Linked MLflow run to case dataset run_id=%s dataset_id=%s dataset_name=%s",
        run_id,
        dataset.dataset_id,
        dataset_name,
    )
    return dataset.dataset_id


def _build_case_dataset_name(
    *,
    case_id: int,
    case_name: str,
    case_payload: dict[str, Any],
    case_source: str,
) -> str:
    """Return one stable MLflow dataset name for an evaluation case."""

    source = slugify(case_source) or "regression"
    payload = json_ready(case_payload)
    profession = slugify(str(payload.get("profession") or case_name))
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
    """Return one case dataset and whether it was newly created."""

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

    payload = json_ready(case_payload)
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
    """Store the current case payload as one dataset record."""

    payload = json_ready(case_payload)
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

    payload = json_ready(case_payload)
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

    value = json_ready(payload)
    if not isinstance(value, dict):
        return value
    return {
        key: item
        for key, item in value.items()
        if key not in _CASE_PAYLOAD_HASH_IGNORED_FIELDS
    }
