"""Public MLflow tracking facade used by app, graph, and eval layers."""

from infrastructure.mlflow.artifacts import (
    log_assistant_response_for_run,
    log_clarification_answer,
    log_clarification_answer_for_run,
    log_clarification_question,
    log_guardrail_blocked_user_message,
    log_json_artifact,
    log_json_artifact_for_run,
    log_llm_prompt_artifact,
    log_metric_for_run,
    log_text_artifact,
    log_text_artifact_for_run,
    set_run_name_for_run,
    set_run_status_for_run,
)
from infrastructure.mlflow.context import (
    MlflowTrackingContext,
    TrackingCapture,
    bind_artifact_namespace,
    bind_mlflow_experiment,
    bind_mlflow_run,
    capture_tracking_events,
    suspend_mlflow_run_termination,
)
from infrastructure.mlflow.eval_datasets import ensure_case_dataset_for_run
from infrastructure.mlflow.runs import (
    activate_mlflow_tracking,
    create_run_id,
    deactivate_mlflow_tracking,
    finalize_mlflow_tracking,
)
from infrastructure.mlflow.tracing import (
    traced_operation,
    update_current_trace_session,
)

__all__ = [
    "MlflowTrackingContext",
    "TrackingCapture",
    "activate_mlflow_tracking",
    "bind_artifact_namespace",
    "bind_mlflow_experiment",
    "bind_mlflow_run",
    "capture_tracking_events",
    "create_run_id",
    "deactivate_mlflow_tracking",
    "ensure_case_dataset_for_run",
    "finalize_mlflow_tracking",
    "log_assistant_response_for_run",
    "log_clarification_answer",
    "log_clarification_answer_for_run",
    "log_clarification_question",
    "log_guardrail_blocked_user_message",
    "log_json_artifact",
    "log_json_artifact_for_run",
    "log_llm_prompt_artifact",
    "log_metric_for_run",
    "log_text_artifact",
    "log_text_artifact_for_run",
    "set_run_name_for_run",
    "set_run_status_for_run",
    "suspend_mlflow_run_termination",
    "traced_operation",
    "update_current_trace_session",
]
