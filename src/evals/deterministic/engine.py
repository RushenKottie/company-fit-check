"""Execution engine for deterministic evaluation cases."""

from __future__ import annotations

from base64 import b64decode
from contextlib import ExitStack
import json
from typing import Any
from unittest.mock import patch

from mlflow.entities.span import SpanType
from pydantic import ValidationError

from evals import repo_root
from evals.deterministic.models import CaseExecutionResult, EvalCase, EvalStubSpec
from evals.deterministic.stubs import STUB_REGISTRY
from graph import nodes as graph_nodes
from graph.node_names import WorkflowNodeName
from application.state_checks import (
    build_case_result_summary,
    validate_company_score_payload,
)
from application.workflow_state import (
    apply_clarification_to_state,
    create_initial_state,
)
from graph.workflow import run_workflow
from models.artifacts import GeneratedArtifact
from models.input import UserInput
from models.state import (
    Axis,
    CompanyCandidate,
    CompanyFitState,
    CompanyScore,
    CompanySearchCriteria,
)
from infrastructure.mlflow_tracking import (
    bind_artifact_namespace,
    bind_mlflow_run,
    capture_tracking_events,
    log_clarification_answer_for_run,
    log_text_artifact,
    suspend_mlflow_run_termination,
    traced_operation,
)
from capabilities.pdf_text import extract_text_from_pdf_bytes
from application.result_exports import build_results_csv

WORKFLOW_TARGETS = {"run_workflow"}
NODE_TARGETS = {
    WorkflowNodeName.EXTRACT_AND_MASK_CV: graph_nodes.extract_and_mask_cv_node,
    WorkflowNodeName.SIMPLIFY_CV: graph_nodes.simplify_cv_node,
    WorkflowNodeName.INTERPRET_USER_INPUT: graph_nodes.interpret_user_input_node,
    WorkflowNodeName.VALIDATE_PII_MASKING: graph_nodes.validate_pii_masking_node,
    WorkflowNodeName.VALIDATE_USER_INPUT_INTERPRETATION: (
        graph_nodes.validate_user_input_interpretation_node
    ),
    WorkflowNodeName.SEARCH_COMPANIES: graph_nodes.search_companies_node,
    WorkflowNodeName.SCORE_COMPANIES: graph_nodes.score_companies_node,
}
HELPER_TARGETS = {
    "score_payload_validator": validate_company_score_payload,
    "build_results_csv": build_results_csv,
}


def execute_case(case: EvalCase, *, run_id: str) -> CaseExecutionResult:
    """Execute one deterministic evaluation case under the given MLflow run id."""

    with bind_mlflow_run(run_id), suspend_mlflow_run_termination(), capture_tracking_events() as capture:
        with bind_artifact_namespace(f"cases/{case.id}"), _apply_stubs(case.setup.stubs):
            with traced_operation(
                "eval.case",
                span_type=SpanType.TASK,
                inputs={
                    "case_id": case.id,
                    "entrypoint_kind": case.entrypoint.kind,
                    "entrypoint_target": case.entrypoint.target,
                },
            ) as span:
                try:
                    result = _execute_case_body(case, run_id=run_id)
                except Exception as exc:
                    formatted_error = _format_exception_message(exc)
                    result = CaseExecutionResult(
                        case_id=case.id,
                        description=case.description,
                        entrypoint_kind=case.entrypoint.kind,
                        entrypoint_target=case.entrypoint.target,
                        run_id=run_id,
                        status="failed",
                        error=formatted_error,
                        uncaught_exception=formatted_error,
                    )
                if span is not None:
                    span.set_outputs(
                        {
                            "status": result.status,
                            "error": result.error,
                            "uncaught_exception": result.uncaught_exception,
                            "executed_span_names": result.executed_span_names,
                        }
                    )
                    if result.uncaught_exception:
                        span.set_attribute("eval.failure_message", result.uncaught_exception)

    result.observed_spans = capture.spans
    result.executed_span_names = [span.get("name", "") for span in capture.spans]
    result.generated_artifacts = capture.artifacts
    return result


def _execute_case_body(case: EvalCase, *, run_id: str) -> CaseExecutionResult:
    """Execute the case body once stubs and MLflow context are already active."""

    if case.entrypoint.kind == "workflow":
        return _execute_workflow_case(case, run_id=run_id)
    if case.entrypoint.kind == "node":
        return _execute_node_case(case, run_id=run_id)
    if case.entrypoint.kind == "helper":
        return _execute_helper_case(case, run_id=run_id)
    raise ValueError(f"Unsupported entrypoint kind: {case.entrypoint.kind}")


def _execute_workflow_case(case: EvalCase, *, run_id: str) -> CaseExecutionResult:
    if case.entrypoint.target not in WORKFLOW_TARGETS:
        raise ValueError(f"Unknown workflow target: {case.entrypoint.target}")

    setup = case.setup
    pdf_bytes = _load_pdf_bytes(setup.pdf_path)
    prompt = setup.prompt or ""
    state = create_initial_state(UserInput(cv_pdf_bytes=pdf_bytes, prompt=prompt), run_id=run_id)
    raw_cv_text = extract_text_from_pdf_bytes(pdf_bytes)
    log_text_artifact("inputs/user-prompt.txt", prompt)

    final_state = run_workflow(state)
    for clarification in case.clarifications:
        log_clarification_answer_for_run(
            run_id=final_state.get("run_id"),
            question=final_state.get("pending_clarification_message"),
            answer=clarification,
            target=final_state.get("clarification_target"),
        )
        final_state = run_workflow(
            apply_clarification_to_state(final_state, clarification)
        )

    csv_artifact = _maybe_log_csv_artifact(final_state)
    return CaseExecutionResult(
        case_id=case.id,
        description=case.description,
        entrypoint_kind=case.entrypoint.kind,
        entrypoint_target=case.entrypoint.target,
        run_id=run_id,
        case_inputs={
            "case_id": case.id,
            "prompt": prompt,
            "raw_cv_text": raw_cv_text,
            "pdf_path": setup.pdf_path,
        },
        final_state_summary=build_case_result_summary(final_state),
        csv_artifact=csv_artifact,
        status=final_state.get("session_status", "failed"),
        error=final_state.get("error"),
    )


def _execute_node_case(case: EvalCase, *, run_id: str) -> CaseExecutionResult:
    try:
        node_target = WorkflowNodeName(case.entrypoint.target)
    except ValueError as exc:
        raise ValueError(f"Unknown node target: {case.entrypoint.target}") from exc

    node = NODE_TARGETS.get(node_target)
    if node is None:
        raise ValueError(f"Unknown node target: {case.entrypoint.target}")

    if case.setup.mode == "workflow_input":
        pdf_bytes = _load_pdf_bytes(case.setup.pdf_path)
        prompt = case.setup.prompt or ""
        state = create_initial_state(
            UserInput(cv_pdf_bytes=pdf_bytes, prompt=prompt),
            run_id=run_id,
        )
        case_inputs = {
            "case_id": case.id,
            "prompt": prompt,
            "raw_cv_text": extract_text_from_pdf_bytes(pdf_bytes),
            "pdf_path": case.setup.pdf_path,
        }
    else:
        state = _load_state_snapshot(case.setup.state_path)
        case_inputs = {
            "case_id": case.id,
            "prompt": state.get("input").prompt if state.get("input") is not None else "",
        }
    state["run_id"] = run_id

    final_state = node(state)
    csv_artifact = _maybe_log_csv_artifact(final_state)
    return CaseExecutionResult(
        case_id=case.id,
        description=case.description,
        entrypoint_kind=case.entrypoint.kind,
        entrypoint_target=case.entrypoint.target,
        run_id=run_id,
        case_inputs=case_inputs,
        final_state_summary=build_case_result_summary(final_state),
        csv_artifact=csv_artifact,
        status=final_state.get("session_status", "failed"),
        error=final_state.get("error"),
    )


def _execute_helper_case(case: EvalCase, *, run_id: str) -> CaseExecutionResult:
    helper = HELPER_TARGETS.get(case.entrypoint.target)
    if helper is None:
        raise ValueError(f"Unknown helper target: {case.entrypoint.target}")

    state = _load_state_snapshot(case.setup.state_path)
    state["run_id"] = run_id
    helper_output = helper(state)
    csv_artifact = None
    if isinstance(helper_output, GeneratedArtifact):
        csv_artifact = _serialize_generated_artifact(helper_output)
        log_text_artifact(
            f"eval/generated/{helper_output.filename}",
            csv_artifact["content"],
        )
    return CaseExecutionResult(
        case_id=case.id,
        description=case.description,
        entrypoint_kind=case.entrypoint.kind,
        entrypoint_target=case.entrypoint.target,
        run_id=run_id,
        case_inputs={"case_id": case.id},
        final_state_summary=build_case_result_summary(state),
        helper_output=helper_output if not isinstance(helper_output, GeneratedArtifact) else None,
        csv_artifact=csv_artifact,
        status=(
            "completed"
            if helper_output is None or helper_output == "" or isinstance(helper_output, GeneratedArtifact)
            else "failed"
        ),
        error=helper_output if isinstance(helper_output, str) and helper_output else None,
    )


def _maybe_log_csv_artifact(state: CompanyFitState) -> dict[str, Any] | None:
    """Build and log one CSV artifact when the state is completed."""

    if state.get("session_status") != "completed":
        return None
    artifact = build_results_csv(state)
    serialized = _serialize_generated_artifact(artifact)
    log_text_artifact(f"eval/generated/{artifact.filename}", serialized["content"])
    return serialized


def _serialize_generated_artifact(artifact: GeneratedArtifact) -> dict[str, Any]:
    """Convert one generated artifact to a JSON-safe payload."""

    return {
        "filename": artifact.filename,
        "content_type": artifact.content_type,
        "content": artifact.content_bytes.decode("utf-8"),
    }


def _load_pdf_bytes(relative_path: str | None) -> bytes:
    if not relative_path:
        raise ValueError("workflow_input setup requires pdf_path.")
    path = (repo_root() / relative_path).resolve()
    return path.read_bytes()


def _load_state_snapshot(relative_path: str | None) -> CompanyFitState:
    if not relative_path:
        raise ValueError("state_snapshot setup requires state_path.")
    path = (repo_root() / relative_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _hydrate_state(payload)


def _hydrate_state(payload: dict[str, Any]) -> CompanyFitState:
    state: CompanyFitState = {}
    if "input" in payload and payload["input"] is not None:
        input_payload = payload["input"]
        if input_payload.get("pdf_path"):
            cv_pdf_bytes = _load_pdf_bytes(input_payload["pdf_path"])
        elif input_payload.get("cv_pdf_bytes_b64"):
            cv_pdf_bytes = b64decode(input_payload["cv_pdf_bytes_b64"])
        else:
            cv_pdf_bytes = b""
        state["input"] = UserInput(
            cv_pdf_bytes=cv_pdf_bytes,
            prompt=input_payload.get("prompt", ""),
        )
    if "masked_cv_text" in payload:
        state["masked_cv_text"] = payload["masked_cv_text"]
    if "pii_masking_status" in payload:
        state["pii_masking_status"] = payload["pii_masking_status"]
    if "simplified_cv_text" in payload:
        state["simplified_cv_text"] = payload["simplified_cv_text"]
    if "company_search_criteria" in payload:
        state["company_search_criteria"] = CompanySearchCriteria.model_validate(
            payload["company_search_criteria"]
        )
    if "axes" in payload:
        state["axes"] = [Axis.model_validate(axis) for axis in payload["axes"]]
    if "companies" in payload:
        state["companies"] = [
            CompanyCandidate.model_validate(company) for company in payload["companies"]
        ]
    if "company_scores" in payload:
        state["company_scores"] = [
            CompanyScore.model_validate(score) for score in payload["company_scores"]
        ]
    for field in [
        "pending_clarification_message",
        "latest_clarification_response",
        "clarification_target",
        "latest_user_message_text",
        "latest_user_message_kind",
        "guardrail_rephrase_source",
        "run_id",
        "user_input_interpretation_clarification_iterations",
        "company_search_clarification_iterations",
        "session_status",
        "error",
    ]:
        if field in payload:
            state[field] = payload[field]
    if "run_id" not in state:
        state["run_id"] = payload.get("mlflow_run_id") or payload.get("debug_session_id")
    return state


def _apply_stubs(stubs: list[EvalStubSpec]):
    stack = ExitStack()
    for spec in stubs:
        stub = STUB_REGISTRY[spec.stub]
        stack.enter_context(patch(spec.target, stub))
    return stack


def _format_exception_message(exc: Exception) -> str:
    """Return one human-readable message for assertion logging."""

    if isinstance(exc, ValidationError):
        first_error = exc.errors()[0] if exc.errors() else {}
        location_parts = tuple(first_error.get("loc", ()))
        location = " -> ".join(str(part) for part in location_parts)
        value = first_error.get("input")
        reason = first_error.get("msg") or str(exc)
        custom_message = _format_validation_error(location_parts, value, reason)
        if custom_message is not None:
            return custom_message
        if location:
            return (
                f"Invalid test fixture or result data at '{location}': "
                f"{reason}. Received value: {value!r}."
            )
        return f"Invalid test fixture or result data: {reason}."

    message = str(exc).strip()
    if not message:
        return f"{exc.__class__.__name__} was raised without an error message."
    return f"{exc.__class__.__name__}: {message}"


def _format_validation_error(
    location_parts: tuple[Any, ...],
    value: Any,
    reason: str,
) -> str | None:
    """Translate common validation errors into clearer QA-facing messages."""

    if (
        len(location_parts) >= 3
        and location_parts[0] == "axis_scores"
        and isinstance(location_parts[1], int)
        and location_parts[2] == "percentage"
    ):
        axis_position = _ordinal(location_parts[1] + 1)
        return (
            f"The score for the {axis_position} axis is invalid: "
            f"it is {value}, but scores must be between 0 and 100."
        )

    if location_parts == ("overall_score",):
        return (
            f"The overall company score is invalid: it is {value}, "
            "but scores must be between 0 and 100."
        )

    if location_parts == ("axes",):
        return (
            "The test data contains an invalid axes list. "
            f"Details: {reason}."
        )

    if location_parts == ("company_search_criteria",):
        return (
            "The test data contains invalid company search criteria. "
            f"Details: {reason}."
        )

    return None


def _ordinal(value: int) -> str:
    """Return one integer as a human-readable ordinal."""

    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"
