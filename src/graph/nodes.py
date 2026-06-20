"""LangGraph node adapters for the application workflow actions."""

from collections.abc import Callable
from typing import Any

from mlflow.entities.span import SpanType

from application.workflow_artifacts import (
    WorkflowRecorder,
    bind_workflow_recorder,
)
from application.workflow_actions import (
    extract_and_mask_cv,
    interpret_user_input_action,
    prepare_final_results_action,
    score_companies_action,
    search_companies_action,
    simplify_cv,
    validate_pii_masking,
    validate_user_input_interpretation_action,
)
from graph.node_names import WorkflowNodeName
from logging_utils import get_logger
from models.state import SESSION_STATUS_FAILED, CompanyFitState
from infrastructure.mlflow_tracking import (
    log_clarification_question,
    log_guardrail_blocked_user_message,
    log_json_artifact,
    log_text_artifact,
    traced_operation,
)

logger = get_logger(__name__)
Action = Callable[[CompanyFitState], CompanyFitState]
InputsBuilder = Callable[[CompanyFitState], dict[str, Any]]


def extract_and_mask_cv_node(state: CompanyFitState) -> CompanyFitState:
    """Run CV text extraction and PII masking as a traced graph node."""

    return _run_action_node(
        state,
        name=WorkflowNodeName.EXTRACT_AND_MASK_CV.value,
        span_type=SpanType.TASK,
        inputs=lambda current: {"session_status": current.get("session_status")},
        action=extract_and_mask_cv,
    )


def validate_pii_masking_node(state: CompanyFitState) -> CompanyFitState:
    """Validate the PII masking result and stop the workflow on masking failure."""

    return _run_action_node(
        state,
        name=WorkflowNodeName.VALIDATE_PII_MASKING.value,
        span_type=SpanType.GUARDRAIL,
        inputs=lambda current: {
            "pii_masking_status": current.get("pii_masking_status")
        },
        action=validate_pii_masking,
    )


def simplify_cv_node(state: CompanyFitState) -> CompanyFitState:
    """Create a simplified CV summary from masked CV text."""

    return _run_action_node(
        state,
        name=WorkflowNodeName.SIMPLIFY_CV.value,
        span_type=SpanType.CHAIN,
        inputs=lambda current: {
            "masked_cv_chars": len(current.get("masked_cv_text") or "")
        },
        action=simplify_cv,
    )


def interpret_user_input_node(state: CompanyFitState) -> CompanyFitState:
    """Interpret the user's prompt and clarifications into search criteria and axes."""

    return _run_action_node(
        state,
        name=WorkflowNodeName.INTERPRET_USER_INPUT.value,
        span_type=SpanType.CHAIN,
        inputs=lambda current: {
            "prompt": current["input"].prompt,
            "clarification_present": bool(
                current.get("latest_clarification_response")
            ),
            "previous_axes": [
                axis.model_dump() for axis in current.get("axes", [])
            ],
        },
        action=interpret_user_input_action,
    )


def validate_user_input_interpretation_node(state: CompanyFitState) -> CompanyFitState:
    """Validate interpreted user intent and request clarification when needed."""

    return _run_action_node(
        state,
        name=WorkflowNodeName.VALIDATE_USER_INPUT_INTERPRETATION.value,
        span_type=SpanType.GUARDRAIL,
        inputs=lambda current: {
            "axes": [axis.model_dump() for axis in current.get("axes", [])],
            "clarification_present": bool(
                current.get("latest_clarification_response")
            ),
        },
        action=validate_user_input_interpretation_action,
    )


def search_companies_node(state: CompanyFitState) -> CompanyFitState:
    """Search for candidate companies using the current company search criteria."""

    return _run_action_node(
        state,
        name=WorkflowNodeName.SEARCH_COMPANIES.value,
        span_type=SpanType.RETRIEVER,
        inputs=lambda current: {
            "company_search_criteria": (
                current["company_search_criteria"].model_dump()
                if current.get("company_search_criteria") is not None
                else None
            )
        },
        action=search_companies_action,
    )


def score_companies_node(state: CompanyFitState) -> CompanyFitState:
    """Score candidate companies against the interpreted user axes."""

    return _run_action_node(
        state,
        name=WorkflowNodeName.SCORE_COMPANIES.value,
        span_type=SpanType.EVALUATOR,
        inputs=lambda current: {
            "company_count": len(current.get("companies", [])),
            "axes": [axis.model_dump() for axis in current.get("axes", [])],
        },
        action=score_companies_action,
    )


def prepare_final_results_node(state: CompanyFitState) -> CompanyFitState:
    """Prepare final ranked company results after scoring completes."""

    return _run_action_node(
        state,
        name=WorkflowNodeName.PREPARE_FINAL_RESULTS.value,
        span_type=SpanType.TASK,
        inputs=lambda current: {"score_count": len(current.get("company_scores", []))},
        action=prepare_final_results_action,
    )


def _run_action_node(
    state: CompanyFitState,
    *,
    name: str,
    span_type: str,
    inputs: InputsBuilder,
    action: Action,
) -> CompanyFitState:
    """Run one workflow action as a logged and traced graph node."""

    logger.info("Node start: %s", name)
    state = _run_traced_action_node(
        state,
        name=name,
        span_type=span_type,
        inputs=inputs,
        action=action,
    )
    logger.info("Node end: %s status=%s", name, state.get("session_status"))
    return state


def _run_traced_action_node(
    state: CompanyFitState,
    *,
    name: str,
    span_type: str,
    inputs: InputsBuilder,
    action: Action,
) -> CompanyFitState:
    """Run one node action inside a trace span and attach recorded outputs."""

    with traced_operation(
        f"node.{name}",
        span_type=span_type,
        inputs=inputs(state),
    ) as span:
        if state.get("session_status") == SESSION_STATUS_FAILED:
            logger.info("Node skip: %s because status=failed", name)
            _set_span_outputs(span, state)
            return state

        state, recorder = _run_recorded_action(state, action)
        _persist_recorded_workflow_observability(recorder, state)
        _set_span_outputs(span, state, **recorder.span_outputs)
        return state


def _run_recorded_action(
    state: CompanyFitState,
    action: Action,
) -> tuple[CompanyFitState, WorkflowRecorder]:
    """Run one action while collecting workflow observability events."""

    recorder = WorkflowRecorder()
    with bind_workflow_recorder(recorder):
        state = action(state)
    return state, recorder


def _persist_recorded_workflow_observability(
    recorder: WorkflowRecorder,
    state: CompanyFitState,
) -> None:
    """Persist observability recorded during an application action."""

    for artifact in recorder.text_artifacts:
        log_text_artifact(artifact.artifact_path, artifact.content)
    for artifact in recorder.json_artifacts:
        log_json_artifact(artifact.artifact_path, artifact.payload)
    for question in recorder.clarification_questions:
        log_clarification_question(question.message, target=question.target)
    for blocked_message in recorder.guardrail_blocked_messages:
        log_guardrail_blocked_user_message(
            run_id=state.get("run_id"),
            message=blocked_message.message,
            source=blocked_message.source,
        )


def _set_span_outputs(span, state: CompanyFitState, **extra) -> None:
    """Attach node result summary and recorded outputs to the trace span."""

    if span is None:
        return
    payload = {
        "session_status": state.get("session_status"),
        "error": state.get("error"),
        "clarification_target": state.get("clarification_target"),
    }
    payload.update(extra)
    span.set_outputs(payload)
