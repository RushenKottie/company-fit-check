"""Application workflow actions used by the LangGraph adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from application.state_checks import (
    validate_company_score_payload,
    validate_search_prerequisites,
)
from llm.client import (
    build_cv_cleanup_due_to_guardrail_message,
    build_generic_llm_failure_message,
    build_rephrase_due_to_guardrail_message,
    build_restart_due_to_guardrail_message,
    is_guardrail_4xx_error,
)
from logging_utils import get_logger
from models.state import CompanyFitState, UserInputInterpretation
from capabilities.company_discovery import (
    build_hard_filter_clarification_message,
    build_zero_company_clarification_message,
    discover_companies,
    get_result_count_hard_filter_violation,
    refine_company_search_criteria,
)
from capabilities.company_scoring import score_companies
from capabilities.cv_simplification import simplify_masked_cv
from capabilities.pdf_text import extract_text_from_pdf_bytes
from capabilities.pii_masking import mask_pii_locally
from capabilities.user_input_interpretation import interpret_user_input
from capabilities.user_input_interpretation_validation import (
    validate_user_input_interpretation,
)

from application.workflow_policy import (
    COMPANY_SEARCH_CLARIFICATION_EXHAUSTED_MESSAGE,
    USER_INPUT_CLARIFICATION_EXHAUSTED_MESSAGE,
    company_search_clarification_limit_reached,
    guardrail_blocked_message_source,
    guardrail_blocked_message_text,
    guardrail_rephrase_exhausted_message,
    user_input_clarification_limit_reached,
)
from application.workflow_transitions import (
    clear_clarification,
    increment_user_input_clarification_iterations,
    mark_completed,
    mark_failed,
    mark_running,
    request_clarification,
    reset_company_search_clarification_iterations,
    reset_user_input_clarification_iterations,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class TextArtifactRequest:
    """Text artifact requested by a workflow action."""

    artifact_path: str
    content: str


@dataclass(slots=True)
class JsonArtifactRequest:
    """JSON artifact requested by a workflow action."""

    artifact_path: str
    payload: Any


@dataclass(slots=True)
class ClarificationQuestionRequest:
    """Clarification question requested by a workflow action."""

    message: str
    target: str


@dataclass(slots=True)
class GuardrailBlockedMessageRequest:
    """Guardrail-blocked user message requested by a workflow action."""

    message: str
    source: str


@dataclass(slots=True)
class WorkflowActionResult:
    """State plus side-effect requests produced by one workflow action."""

    state: CompanyFitState
    span_outputs: dict[str, Any] = field(default_factory=dict)
    text_artifacts: list[TextArtifactRequest] = field(default_factory=list)
    json_artifacts: list[JsonArtifactRequest] = field(default_factory=list)
    clarification_questions: list[ClarificationQuestionRequest] = field(default_factory=list)
    guardrail_blocked_messages: list[GuardrailBlockedMessageRequest] = field(default_factory=list)


def extract_and_mask_cv(state: CompanyFitState) -> WorkflowActionResult:
    """Extract CV text and mask PII before any LLM step."""

    try:
        extracted_cv_text = extract_text_from_pdf_bytes(state["input"].cv_pdf_bytes)
    except Exception as exc:
        logger.exception("PDF text extraction failed.")
        mark_failed(state, f"PDF text extraction failed: {exc}")
        return WorkflowActionResult(state)

    try:
        result = mask_pii_locally(extracted_cv_text)
    except Exception as exc:
        logger.exception("PII masking failed.")
        mark_failed(state, f"PII masking failed: {exc}")
        state["pii_masking_status"] = "failed"
        return WorkflowActionResult(state)

    state["masked_cv_text"] = result.text
    state["pii_masking_status"] = "passed"
    return WorkflowActionResult(
        state=state,
        span_outputs={
            "masked_cv_chars": len(result.text),
            "pii_masking_status": state.get("pii_masking_status"),
        },
        text_artifacts=[
            TextArtifactRequest("workflow/masked-cv.txt", result.text),
        ],
    )


def validate_pii_masking(state: CompanyFitState) -> WorkflowActionResult:
    """Validate that PII masking completed successfully."""

    masked = state.get("masked_cv_text") or ""
    if state.get("pii_masking_status") != "passed":
        mark_failed(state, "PII masking did not complete successfully.")
    elif not masked:
        mark_failed(state, "PII masking produced empty output.")
    return WorkflowActionResult(
        state=state,
        span_outputs={"masked_cv_chars": len(masked)},
    )


def simplify_cv(state: CompanyFitState) -> WorkflowActionResult:
    """Simplify the masked CV text for downstream LLM steps."""

    try:
        state["simplified_cv_text"] = simplify_masked_cv(state["masked_cv_text"] or "")
    except Exception as exc:
        logger.exception("CV simplification failed.")
        if is_guardrail_4xx_error(exc):
            mark_failed(state, build_cv_cleanup_due_to_guardrail_message())
        else:
            mark_failed(state, build_generic_llm_failure_message())

    result = WorkflowActionResult(
        state=state,
        span_outputs={
            "simplified_cv_chars": len(state.get("simplified_cv_text") or "")
        },
    )
    if state.get("simplified_cv_text"):
        result.text_artifacts.append(
            TextArtifactRequest(
                "workflow/simplified-cv.txt",
                state["simplified_cv_text"],
            )
        )
    return result


def interpret_user_input_action(state: CompanyFitState) -> WorkflowActionResult:
    """Interpret user input into search criteria and scoring axes."""

    previous_user_input_interpretation = UserInputInterpretation(
        company_search_criteria=state["company_search_criteria"],
        axes=state.get("axes", []),
    )
    try:
        user_input_interpretation = interpret_user_input(
            prompt=state["input"].prompt,
            simplified_cv_text=state["simplified_cv_text"] or "",
            clarification=state.get("latest_clarification_response"),
            previous_axes=state.get("axes", []),
        )
    except Exception as exc:
        logger.exception("User-input interpretation failed.")
        if is_guardrail_4xx_error(exc):
            return _request_user_input_rephrase_after_guardrail(state)
        mark_failed(state, build_generic_llm_failure_message())
        return WorkflowActionResult(state)

    state["company_search_criteria"] = user_input_interpretation.company_search_criteria
    state["axes"] = user_input_interpretation.axes
    if user_input_interpretation != previous_user_input_interpretation:
        reset_user_input_clarification_iterations(state)

    return WorkflowActionResult(
        state=state,
        span_outputs={
            "axes": [axis.model_dump() for axis in user_input_interpretation.axes],
            "company_search_criteria": (
                user_input_interpretation.company_search_criteria.model_dump()
            ),
        },
        json_artifacts=[
            JsonArtifactRequest(
                f"workflow/user-input-interpretation-{_artifact_timestamp()}.json",
                {
                    "company_search_criteria": (
                        user_input_interpretation.company_search_criteria.model_dump()
                    ),
                    "axes": [
                        axis.model_dump() for axis in user_input_interpretation.axes
                    ],
                },
            )
        ],
    )


def validate_user_input_interpretation_action(
    state: CompanyFitState,
) -> WorkflowActionResult:
    """Validate interpreted axes and request clarification when needed."""

    try:
        valid, message = validate_user_input_interpretation(
            axes=state.get("axes", []),
            simplified_cv_text=state.get("simplified_cv_text") or "",
            prompt=state["input"].prompt,
            clarification=state.get("latest_clarification_response"),
        )
    except Exception as exc:
        logger.exception("User-input interpretation validation failed.")
        if is_guardrail_4xx_error(exc):
            result = _request_user_input_rephrase_after_guardrail(state)
            if result.state.get("session_status") == "needs_clarification":
                result.span_outputs["valid"] = False
            return result
        mark_failed(state, build_generic_llm_failure_message())
        return WorkflowActionResult(state)

    if valid:
        clear_clarification(state)
        reset_user_input_clarification_iterations(state)
        mark_running(state)
        state["latest_clarification_response"] = None
        return WorkflowActionResult(
            state=state,
            span_outputs={"valid": valid, "clarification_message": message},
        )

    if user_input_clarification_limit_reached(state):
        mark_failed(state, USER_INPUT_CLARIFICATION_EXHAUSTED_MESSAGE)
        return WorkflowActionResult(
            state=state,
            span_outputs={"valid": valid, "clarification_message": message},
        )

    request_clarification(
        state,
        target="user_input_interpretation",
        message=message,
    )
    result = WorkflowActionResult(
        state=state,
        span_outputs={"valid": valid, "clarification_message": message},
    )
    if message:
        result.clarification_questions.append(
            ClarificationQuestionRequest(message, "user_input_interpretation")
        )
    return result


def search_companies_action(state: CompanyFitState) -> WorkflowActionResult:
    """Discover companies and handle discovery clarifications."""

    prerequisite_error = validate_search_prerequisites(state)
    if prerequisite_error:
        mark_failed(state, prerequisite_error)
        return WorkflowActionResult(
            state=state,
            span_outputs={"prerequisite_error": prerequisite_error},
        )

    try:
        state["companies"] = discover_companies(
            company_search_criteria=state["company_search_criteria"],
        )
    except Exception as exc:
        logger.exception("Company discovery failed.")
        if is_guardrail_4xx_error(exc):
            mark_failed(state, build_restart_due_to_guardrail_message("company search"))
        else:
            mark_failed(state, build_generic_llm_failure_message())
        return WorkflowActionResult(state)

    hard_filter_violation = get_result_count_hard_filter_violation(
        state["company_search_criteria"],
        len(state["companies"]),
    )
    if hard_filter_violation is not None:
        if company_search_clarification_limit_reached(state):
            mark_failed(state, COMPANY_SEARCH_CLARIFICATION_EXHAUSTED_MESSAGE)
            return WorkflowActionResult(
                state=state,
                span_outputs={
                    "company_count": len(state["companies"]),
                    "hard_filter": hard_filter_violation.model_dump(),
                },
            )

        message = build_hard_filter_clarification_message(
            hard_filter_violation,
            len(state["companies"]),
        )
        request_clarification(state, target="company_search", message=message)
        return WorkflowActionResult(
            state=state,
            span_outputs={
                "company_count": len(state["companies"]),
                "clarification_message": state.get("pending_clarification_message"),
                "hard_filter": hard_filter_violation.model_dump(),
            },
            clarification_questions=[
                ClarificationQuestionRequest(message, "company_search")
            ],
        )

    if not state["companies"]:
        if company_search_clarification_limit_reached(state):
            mark_failed(state, COMPANY_SEARCH_CLARIFICATION_EXHAUSTED_MESSAGE)
            return WorkflowActionResult(
                state=state,
                span_outputs={"company_count": 0},
            )
        try:
            message = build_zero_company_clarification_message(
                state["company_search_criteria"]
            )
        except Exception as exc:
            logger.exception("Zero-result clarification failed.")
            if is_guardrail_4xx_error(exc):
                mark_failed(state, build_restart_due_to_guardrail_message("company search"))
            else:
                mark_failed(state, build_generic_llm_failure_message())
            return WorkflowActionResult(
                state=state,
                span_outputs={"company_count": 0},
            )

        request_clarification(state, target="company_search", message=message)
        result = WorkflowActionResult(
            state=state,
            span_outputs={
                "company_count": 0,
                "clarification_message": state.get("pending_clarification_message"),
            },
        )
        if message:
            result.clarification_questions.append(
                ClarificationQuestionRequest(message, "company_search")
            )
        return result

    clear_clarification(state)
    reset_company_search_clarification_iterations(state)
    mark_running(state)
    return WorkflowActionResult(
        state=state,
        span_outputs={
            "company_count": len(state["companies"]),
            "companies": [company.model_dump() for company in state["companies"]],
        },
        json_artifacts=[
            JsonArtifactRequest(
                f"workflow/discovered-companies-{_artifact_timestamp()}.json",
                [company.model_dump() for company in state["companies"]],
            )
        ],
    )


def score_companies_action(state: CompanyFitState) -> WorkflowActionResult:
    """Score discovered companies against the interpreted axes."""

    try:
        state["company_scores"] = score_companies(
            companies=state.get("companies", []),
            simplified_cv_text=state.get("simplified_cv_text") or "",
            axes=state.get("axes", []),
        )
    except Exception as exc:
        logger.exception("Company scoring failed.")
        if is_guardrail_4xx_error(exc):
            mark_failed(state, build_restart_due_to_guardrail_message("company scoring"))
        else:
            mark_failed(state, build_generic_llm_failure_message())
        return WorkflowActionResult(state)

    payload_error = validate_company_score_payload(state)
    if payload_error:
        mark_failed(state, payload_error)
        return WorkflowActionResult(
            state=state,
            span_outputs={"payload_error": payload_error},
        )

    mark_completed(state)
    return WorkflowActionResult(
        state=state,
        span_outputs={
            "score_count": len(state["company_scores"]),
            "company_scores": [score.model_dump() for score in state["company_scores"]],
        },
        json_artifacts=[
            JsonArtifactRequest(
                "workflow/company-scores.json",
                [score.model_dump() for score in state["company_scores"]],
            )
        ],
    )


def refine_company_search_action(state: CompanyFitState) -> WorkflowActionResult:
    """Refine company search criteria from a user clarification."""

    clarification = state.get("latest_clarification_response")
    if not clarification:
        mark_failed(state, "Missing company-search clarification response.")
        return WorkflowActionResult(state)

    try:
        previous_criteria = state["company_search_criteria"]
        state["company_search_criteria"] = refine_company_search_criteria(
            company_search_criteria=state["company_search_criteria"],
            user_clarification=clarification,
        )
    except Exception as exc:
        logger.exception("Company-search refinement failed.")
        if is_guardrail_4xx_error(exc):
            mark_failed(
                state,
                build_restart_due_to_guardrail_message("company search refinement"),
            )
        else:
            mark_failed(state, build_generic_llm_failure_message())
        return WorkflowActionResult(state)

    if state["company_search_criteria"] != previous_criteria:
        reset_company_search_clarification_iterations(state)
    state["latest_clarification_response"] = None
    clear_clarification(state)
    mark_running(state)
    return WorkflowActionResult(
        state=state,
        span_outputs={
            "company_search_criteria": state["company_search_criteria"].model_dump()
        },
        json_artifacts=[
            JsonArtifactRequest(
                f"workflow/refined-company-search-{_artifact_timestamp()}.json",
                state["company_search_criteria"].model_dump(),
            )
        ],
    )


def _request_user_input_rephrase_after_guardrail(
    state: CompanyFitState,
) -> WorkflowActionResult:
    """Move the workflow into clarification mode after a recoverable guardrail failure."""

    if user_input_clarification_limit_reached(state):
        mark_failed(state, guardrail_rephrase_exhausted_message())
        return WorkflowActionResult(state)

    blocked_source = guardrail_blocked_message_source(state)
    blocked_message = guardrail_blocked_message_text(state)
    message = build_rephrase_due_to_guardrail_message()
    request_clarification(
        state,
        target="user_input_interpretation",
        message=message,
    )
    state["error"] = None
    state["guardrail_rephrase_source"] = blocked_source
    if blocked_source == "prompt":
        state["input"].prompt = ""
    else:
        state["latest_clarification_response"] = None
    state["latest_user_message_text"] = None
    state["latest_user_message_kind"] = None
    increment_user_input_clarification_iterations(state)

    result = WorkflowActionResult(
        state=state,
        span_outputs={
            "guardrail_rephrase_requested": True,
            "clarification_message": state.get("pending_clarification_message"),
        },
        clarification_questions=[
            ClarificationQuestionRequest(message, "user_input_interpretation")
        ],
    )
    if blocked_message:
        result.guardrail_blocked_messages.append(
            GuardrailBlockedMessageRequest(blocked_message, blocked_source)
        )
    return result


def _artifact_timestamp() -> str:
    """Return a compact timestamp for one artifact filename."""

    return str(int(perf_counter() * 1000))
