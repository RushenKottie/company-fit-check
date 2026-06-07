"""LangGraph node adapters for the application workflow actions."""

from collections.abc import Callable
from time import perf_counter
from typing import Any

from mlflow.entities.span import SpanType

from application.workflow_actions import (
    WorkflowActionResult,
    extract_and_mask_cv,
    interpret_user_input_action,
    refine_company_search_action,
    score_companies_action,
    search_companies_action,
    simplify_cv,
    validate_pii_masking,
    validate_user_input_interpretation_action,
)
from graph.node_names import WorkflowNodeName
from logging_utils import get_logger
from models.state import CompanyFitState
from infrastructure.mlflow_tracking import (
    log_clarification_question,
    log_guardrail_blocked_user_message,
    log_json_artifact,
    log_text_artifact,
    traced_operation,
)

logger = get_logger(__name__)
Action = Callable[[CompanyFitState], WorkflowActionResult]
InputsBuilder = Callable[[CompanyFitState], dict[str, Any]]


def extract_and_mask_cv_node(state: CompanyFitState) -> CompanyFitState:
    return _run_action_node(
        state,
        name=WorkflowNodeName.EXTRACT_AND_MASK_CV.value,
        span_type=SpanType.TASK,
        inputs=lambda current: {"session_status": current.get("session_status")},
        action=extract_and_mask_cv,
    )


def validate_pii_masking_node(state: CompanyFitState) -> CompanyFitState:
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


def refine_company_search_node(state: CompanyFitState) -> CompanyFitState:
    return _run_action_node(
        state,
        name=WorkflowNodeName.REFINE_COMPANY_SEARCH.value,
        span_type=SpanType.CHAIN,
        inputs=lambda current: {
            "company_search_criteria": current["company_search_criteria"].model_dump(),
            "clarification": current.get("latest_clarification_response"),
        },
        action=refine_company_search_action,
    )


def _run_action_node(
    state: CompanyFitState,
    *,
    name: str,
    span_type: str,
    inputs: InputsBuilder,
    action: Action,
) -> CompanyFitState:
    start = perf_counter()
    logger.info("Node start: %s", name)
    with traced_operation(
        f"node.{name}",
        span_type=span_type,
        inputs=inputs(state),
    ) as span:
        if state.get("session_status") == "failed":
            logger.info("Node skip: %s because status=failed", name)
            _set_span_outputs(span, state)
            return state

        result = action(state)
        _emit_action_side_effects(result)
        _set_span_outputs(span, result.state, **result.span_outputs)

    logger.info(
        "Node end: %s status=%s duration_ms=%.1f",
        name,
        state.get("session_status"),
        (perf_counter() - start) * 1000,
    )
    return state


def _emit_action_side_effects(result: WorkflowActionResult) -> None:
    """Persist tracking side effects requested by an application action."""

    for artifact in result.text_artifacts:
        log_text_artifact(artifact.artifact_path, artifact.content)
    for artifact in result.json_artifacts:
        log_json_artifact(artifact.artifact_path, artifact.payload)
    for question in result.clarification_questions:
        log_clarification_question(question.message, target=question.target)
    for blocked_message in result.guardrail_blocked_messages:
        log_guardrail_blocked_user_message(
            run_id=result.state.get("run_id"),
            message=blocked_message.message,
            source=blocked_message.source,
        )


def _set_span_outputs(span, state: CompanyFitState, **extra) -> None:
    """Attach the current node result summary to the active MLflow span."""

    if span is None:
        return
    payload = {
        "session_status": state.get("session_status"),
        "error": state.get("error"),
        "clarification_target": state.get("clarification_target"),
    }
    payload.update(extra)
    span.set_outputs(payload)
