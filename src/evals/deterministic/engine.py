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
from application.state_checks import validate_company_score_payload
from application.workflow_state import (
    apply_clarification_to_state,
    create_initial_state,
)
from graph.workflow import run_workflow
from models.artifacts import GeneratedCsvArtifact
from models.input import UserInput
from models.state import (
    SESSION_STATUS_COMPLETED,
    SESSION_STATUS_FAILED,
    Axis,
    CompanyCandidate,
    CompanyFitState,
    CompanyScore,
    CompanySearchCriteria,
    FinalCompanyResult,
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
from services.pdf_text import extract_text_from_pdf_bytes
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
NODE_SPAN_PREFIX = "node."


def build_case_result_summary(state: CompanyFitState) -> dict[str, Any]:
    """Return the state fields needed by deterministic checks and reports."""

    return {
        "input": (
            {"prompt": state["input"].prompt}
            if state.get("input") is not None
            else None
        ),
        "session_status": state.get("session_status"),
        "error": state.get("error"),
        "pii_masking_status": state.get("pii_masking_status"),
        "clarification_target": state.get("clarification_target"),
        "user_input_interpretation_clarification_iterations": state.get(
            "user_input_interpretation_clarification_iterations",
            0,
        ),
        "masked_cv_text": state.get("masked_cv_text"),
        "simplified_cv_text": state.get("simplified_cv_text"),
        "axes": [axis.model_dump() for axis in state.get("axes", [])],
        "company_search_criteria": (
            state["company_search_criteria"].model_dump()
            if state.get("company_search_criteria") is not None
            else None
        ),
        "companies": [company.model_dump() for company in state.get("companies", [])],
        "company_scores": [
            score.model_dump() for score in state.get("company_scores", [])
        ],
        "final_results": [
            result.model_dump() for result in state.get("final_results", [])
        ],
        "pending_clarification_message": state.get("pending_clarification_message"),
        "latest_user_message_text": state.get("latest_user_message_text"),
        "latest_user_message_kind": state.get("latest_user_message_kind"),
        "guardrail_rephrase_source": state.get("guardrail_rephrase_source"),
    }


def execute_case(case: EvalCase, *, run_id: str) -> CaseExecutionResult:
    """Run one deterministic case and attach captured nodes, spans, and artifacts."""

    with (
        bind_mlflow_run(run_id),
        suspend_mlflow_run_termination(),
        capture_tracking_events() as capture,
    ):
        with bind_artifact_namespace(f"cases/{case.id}"), _apply_stubs(case.setup.stubs):
            result = _execute_traced_case(
                case,
                run_id=run_id,
                capture=capture,
            )

    _attach_captured_outputs(result, capture)
    return result


def _execute_traced_case(
    case: EvalCase,
    *,
    run_id: str,
    capture: Any,
) -> CaseExecutionResult:
    """Run a case inside the MLflow eval span and record span outputs."""

    with traced_operation(
        "eval.case",
        span_type=SpanType.TASK,
        inputs={
            "case_id": case.id,
            "entrypoint_kind": case.entrypoint.kind,
            "entrypoint_target": case.entrypoint.target,
        },
    ) as span:
        result = _execute_case_or_failure(case, run_id=run_id)
        result.executed_node_names = _captured_node_names(capture)
        _record_case_span_outputs(span, result)
        return result


def _execute_case_or_failure(case: EvalCase, *, run_id: str) -> CaseExecutionResult:
    """Run a case body or convert its exception into a failed result."""

    try:
        return _execute_case_body(case, run_id=run_id)
    except Exception as exc:
        formatted_error = _format_exception_message(exc)
        return CaseExecutionResult(
            case_id=case.id,
            description=case.description,
            entrypoint_kind=case.entrypoint.kind,
            entrypoint_target=case.entrypoint.target,
            run_id=run_id,
            status=SESSION_STATUS_FAILED,
            error=formatted_error,
            uncaught_exception=formatted_error,
        )


def _record_case_span_outputs(span: Any, result: CaseExecutionResult) -> None:
    """Write case status, errors, and executed nodes to the MLflow span."""

    if span is None:
        return

    span.set_outputs(
        {
            "status": result.status,
            "error": result.error,
            "uncaught_exception": result.uncaught_exception,
            "executed_node_names": result.executed_node_names,
        }
    )
    if result.uncaught_exception:
        span.set_attribute(
            "eval.failure_message",
            result.uncaught_exception,
        )


def _attach_captured_outputs(
    result: CaseExecutionResult,
    capture: Any,
) -> None:
    """Attach captured nodes, MLflow spans, and artifacts to a case result."""

    result.executed_node_names = _captured_node_names(capture)
    result.observed_mlflow_spans = capture.spans
    result.executed_mlflow_span_names = [
        span.get("name", "") for span in capture.spans
    ]
    result.generated_artifacts = capture.artifacts


def _captured_node_names(capture: Any) -> list[str]:
    """Return workflow node names captured from MLflow node spans."""

    return [
        name.removeprefix(NODE_SPAN_PREFIX)
        for span in capture.spans
        if (name := span.get("name", "")).startswith(NODE_SPAN_PREFIX)
    ]


def _execute_case_body(case: EvalCase, *, run_id: str) -> CaseExecutionResult:
    """Dispatch a case to the configured workflow, node, or helper entrypoint."""

    if case.entrypoint.kind == "workflow":
        return _execute_workflow_case(case, run_id=run_id)
    if case.entrypoint.kind == "node":
        return _execute_node_case(case, run_id=run_id)
    if case.entrypoint.kind == "helper":
        return _execute_helper_case(case, run_id=run_id)
    raise ValueError(f"Unsupported entrypoint kind: {case.entrypoint.kind}")


def _execute_workflow_case(case: EvalCase, *, run_id: str) -> CaseExecutionResult:
    """Run a full workflow case from its PDF, prompt, and clarification steps."""

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

    csv_artifact = _build_and_log_csv_for_completed_state(final_state)
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
        status=final_state.get("session_status", SESSION_STATUS_FAILED),
        error=final_state.get("error"),
    )


def _execute_node_case(case: EvalCase, *, run_id: str) -> CaseExecutionResult:
    """Run one workflow node from either fresh input or a saved state fixture."""

    try:
        node_target = WorkflowNodeName(case.entrypoint.target)
    except ValueError as exc:
        raise ValueError(f"Unknown node target: {case.entrypoint.target}") from exc

    node = NODE_TARGETS.get(node_target)
    if node is None:
        raise ValueError(f"Unknown node target: {case.entrypoint.target}")

    state, case_inputs = _build_node_case_state(case, run_id=run_id)
    final_state = node(state)
    csv_artifact = _build_and_log_csv_for_completed_state(final_state)
    return CaseExecutionResult(
        case_id=case.id,
        description=case.description,
        entrypoint_kind=case.entrypoint.kind,
        entrypoint_target=case.entrypoint.target,
        run_id=run_id,
        case_inputs=case_inputs,
        final_state_summary=build_case_result_summary(final_state),
        csv_artifact=csv_artifact,
        status=final_state.get("session_status", SESSION_STATUS_FAILED),
        error=final_state.get("error"),
    )


def _build_node_case_state(
    case: EvalCase,
    *,
    run_id: str,
) -> tuple[CompanyFitState, dict[str, Any]]:
    """Build the starting state and check inputs for a node case."""

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
    return state, case_inputs


def _execute_helper_case(case: EvalCase, *, run_id: str) -> CaseExecutionResult:
    """Run a standalone eval target"""

    helper = HELPER_TARGETS.get(case.entrypoint.target)
    if helper is None:
        raise ValueError(f"Unknown helper target: {case.entrypoint.target}")

    state = _load_state_snapshot(case.setup.state_path)
    state["run_id"] = run_id
    helper_output = helper(state)
    csv_artifact = None
    helper_output_is_csv = isinstance(helper_output, GeneratedCsvArtifact)
    if helper_output_is_csv:
        csv_artifact = _serialize_generated_csv_artifact(helper_output)
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
        helper_output=helper_output if not helper_output_is_csv else None,
        csv_artifact=csv_artifact,
        status=(
            SESSION_STATUS_COMPLETED
            if helper_output is None or helper_output == "" or helper_output_is_csv
            else SESSION_STATUS_FAILED
        ),
        error=helper_output if isinstance(helper_output, str) and helper_output else None,
    )


def _build_and_log_csv_for_completed_state(state: CompanyFitState) -> dict[str, Any] | None:
    """Build and log the results CSV only for a completed workflow state."""

    if state.get("session_status") != SESSION_STATUS_COMPLETED:
        return None
    artifact = build_results_csv(state)
    serialized = _serialize_generated_csv_artifact(artifact)
    log_text_artifact(f"eval/generated/{artifact.filename}", serialized["content"])
    return serialized


def _serialize_generated_csv_artifact(artifact: GeneratedCsvArtifact) -> dict[str, Any]:
    """Convert a generated CSV file into a JSON-safe result payload."""

    return {
        "filename": artifact.filename,
        "content_type": artifact.content_type,
        "content": artifact.content_bytes.decode("utf-8"),
    }


def _load_pdf_bytes(relative_path: str | None) -> bytes:
    """Read the PDF fixture bytes required by a workflow input case."""

    if not relative_path:
        raise ValueError("workflow_input setup requires pdf_path.")
    path = (repo_root() / relative_path).resolve()
    return path.read_bytes()


def _load_state_snapshot(relative_path: str | None) -> CompanyFitState:
    """Load a JSON state fixture and convert it into workflow state."""

    if not relative_path:
        raise ValueError("state_snapshot setup requires state_path.")
    path = (repo_root() / relative_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _build_state_from_fixture(payload)


def _build_state_from_fixture(payload: dict[str, Any]) -> CompanyFitState:
    """Convert fixture data into the typed models used by workflow state."""

    state: CompanyFitState = {}
    input_payload = payload.get("input")
    if input_payload is not None:
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

    for field in (
        "masked_cv_text",
        "pii_masking_status",
        "simplified_cv_text",
        "pending_clarification_message",
        "latest_clarification_response",
        "clarification_target",
        "latest_user_message_text",
        "latest_user_message_kind",
        "guardrail_rephrase_source",
        "run_id",
        "user_input_interpretation_clarification_iterations",
        "session_status",
        "error",
    ):
        if field in payload:
            state[field] = payload[field]

    if "company_search_criteria" in payload:
        state["company_search_criteria"] = CompanySearchCriteria.model_validate(
            payload["company_search_criteria"]
        )

    list_models = {
        "axes": Axis,
        "companies": CompanyCandidate,
        "company_scores": CompanyScore,
        "final_results": FinalCompanyResult,
    }
    for field, model in list_models.items():
        if field in payload:
            state[field] = [model.model_validate(item) for item in payload[field]]

    if "run_id" not in state:
        state["run_id"] = payload.get("mlflow_run_id") or payload.get("debug_session_id")
    return state


def _apply_stubs(stubs: list[EvalStubSpec]):
    """Patch configured targets with deterministic stubs for one case run."""

    stack = ExitStack()
    for spec in stubs:
        stub = STUB_REGISTRY[spec.stub]
        stack.enter_context(patch(spec.target, stub))
    return stack


def _format_exception_message(exc: Exception) -> str:
    """Convert an exception into the failure text shown in eval results."""

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
    """Return clearer messages for validation errors the eval suite expects."""

    score_label: str | None = None
    if location_parts == ("overall_score",):
        score_label = "overall company score"
    elif (
        len(location_parts) >= 3
        and location_parts[0] == "axis_scores"
        and isinstance(location_parts[1], int)
        and location_parts[2] == "percentage"
    ):
        score_label = f"score for the {_ordinal(location_parts[1] + 1)} axis"

    if score_label:
        return (
            f"The {score_label} is invalid: it is {value}, "
            "but scores must be between 0 and 100."
        )

    invalid_fixture_fields = {
        ("axes",): "invalid axes list",
        ("company_search_criteria",): "invalid company search criteria",
    }
    if location_parts in invalid_fixture_fields:
        return (
            f"The test data contains {invalid_fixture_fields[location_parts]}. "
            f"Details: {reason}."
        )

    return None


def _ordinal(value: int) -> str:
    """Format an integer as an ordinal such as 1st, 2nd, or 3rd."""

    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"
