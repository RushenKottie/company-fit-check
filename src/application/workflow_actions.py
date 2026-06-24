"""Application workflow actions that update state for downstream processing."""

from __future__ import annotations

from application.state_checks import (
    validate_company_score_payload,
    validate_search_prerequisites,
)
from application.workflow_artifacts import (
    artifact_timestamp,
    record_clarification_question,
    record_guardrail_blocked_message,
    record_json_artifact,
    record_span_outputs,
    record_text_artifact,
)
from application.messages import (
    build_cv_cleanup_due_to_guardrail_message,
    build_generic_llm_failure_message,
    build_rephrase_due_to_guardrail_message,
    build_restart_due_to_guardrail_message,
)
from llm.client import (
    is_guardrail_4xx_error,
)
from logging_utils import get_logger
from models.state import (
    PII_MASKING_STATUS_FAILED,
    PII_MASKING_STATUS_PASSED,
    CompanyFitState,
    FinalAxisResult,
    FinalCompanyResult,
    UserInputInterpretation,
)
from services.company_discovery import (
    discover_companies,
    get_result_count_maximum,
)
from services.company_scoring import score_companies
from services.cv_simplification import simplify_masked_cv
from services.pdf_text import extract_text_from_pdf_bytes
from services.pii_masking import mask_pii_locally
from services.user_input_interpretation import interpret_user_input
from services.user_input_interpretation_validation import (
    validate_user_input_interpretation,
)

from application.workflow_policy import (
    USER_INPUT_CLARIFICATION_EXHAUSTED_MESSAGE,
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
    reset_user_input_clarification_iterations,
)

logger = get_logger(__name__)


def extract_and_mask_cv(state: CompanyFitState) -> CompanyFitState:
    """Extract CV text and mask PII before any LLM step."""

    try:
        extracted_cv_text = extract_text_from_pdf_bytes(state["input"].cv_pdf_bytes)
    except Exception as exc:
        logger.exception("PDF text extraction failed.")
        mark_failed(state, f"PDF text extraction failed: {exc}")
        return state

    try:
        result = mask_pii_locally(extracted_cv_text)
    except Exception as exc:
        logger.exception("PII masking failed.")
        mark_failed(state, f"PII masking failed: {exc}")
        state["pii_masking_status"] = PII_MASKING_STATUS_FAILED
        return state

    state["masked_cv_text"] = result.text
    state["pii_masking_status"] = PII_MASKING_STATUS_PASSED
    record_span_outputs(
        masked_cv_chars=len(result.text),
        pii_masking_status=PII_MASKING_STATUS_PASSED,
    )
    record_text_artifact("workflow/masked-cv.txt", result.text)
    return state


def validate_pii_masking(state: CompanyFitState) -> CompanyFitState:
    """Validate that PII masking completed successfully."""

    masked = state.get("masked_cv_text") or ""
    if state.get("pii_masking_status") != PII_MASKING_STATUS_PASSED:
        mark_failed(state, "PII masking did not complete successfully.")
    elif not masked:
        mark_failed(state, "PII masking produced empty output.")
    record_span_outputs(masked_cv_chars=len(masked))
    return state


def simplify_cv(state: CompanyFitState) -> CompanyFitState:
    """Simplify the masked CV text for downstream LLM steps."""

    try:
        state["simplified_cv_text"] = simplify_masked_cv(state["masked_cv_text"] or "")
    except Exception as exc:
        logger.exception("CV simplification failed.")
        if is_guardrail_4xx_error(exc):
            mark_failed(state, build_cv_cleanup_due_to_guardrail_message())
        else:
            mark_failed(state, build_generic_llm_failure_message())

    record_span_outputs(
        simplified_cv_chars=len(state.get("simplified_cv_text") or "")
    )
    if state.get("simplified_cv_text"):
        record_text_artifact(
            "workflow/simplified-cv.txt",
            state["simplified_cv_text"],
        )
    return state


def interpret_user_input_action(state: CompanyFitState) -> CompanyFitState:
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
        return state

    state["company_search_criteria"] = user_input_interpretation.company_search_criteria
    state["axes"] = user_input_interpretation.axes
    if user_input_interpretation != previous_user_input_interpretation:
        reset_user_input_clarification_iterations(state)

    interpretation_payload = {
        "company_search_criteria": (
            user_input_interpretation.company_search_criteria.model_dump()
        ),
        "axes": [axis.model_dump() for axis in user_input_interpretation.axes],
    }
    record_span_outputs(**interpretation_payload)
    record_json_artifact(
        f"workflow/user-input-interpretation-{artifact_timestamp()}.json",
        interpretation_payload,
    )
    return state


def validate_user_input_interpretation_action(
    state: CompanyFitState,
) -> CompanyFitState:
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
            return _request_user_input_rephrase_after_guardrail(state)
        mark_failed(state, build_generic_llm_failure_message())
        return state

    if valid:
        clear_clarification(state)
        reset_user_input_clarification_iterations(state)
        mark_running(state)
        state["latest_clarification_response"] = None
        record_span_outputs(valid=valid, clarification_message=message)
        return state

    if user_input_clarification_limit_reached(state):
        mark_failed(state, USER_INPUT_CLARIFICATION_EXHAUSTED_MESSAGE)
        record_span_outputs(valid=valid, clarification_message=message)
        return state

    request_clarification(
        state,
        target="user_input_interpretation",
        message=message,
    )
    record_span_outputs(valid=valid, clarification_message=message)
    if message:
        record_clarification_question(message, "user_input_interpretation")
    return state


def search_companies_action(state: CompanyFitState) -> CompanyFitState:
    """Discover companies from the interpreted search criteria."""

    prerequisite_error = validate_search_prerequisites(state)
    if prerequisite_error:
        mark_failed(state, prerequisite_error)
        record_span_outputs(prerequisite_error=prerequisite_error)
        return state

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
        return state

    company_count = len(state["companies"])
    clear_clarification(state)
    mark_running(state)
    companies_payload = [company.model_dump() for company in state["companies"]]
    record_span_outputs(
        company_count=company_count,
        companies=companies_payload,
    )
    record_json_artifact(
        f"workflow/discovered-companies-{artifact_timestamp()}.json",
        companies_payload,
    )
    return state


def score_companies_action(state: CompanyFitState) -> CompanyFitState:
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
        return state

    payload_error = validate_company_score_payload(state)
    if payload_error:
        mark_failed(state, payload_error)
        record_span_outputs(payload_error=payload_error)
        return state

    scores_payload = [score.model_dump() for score in state["company_scores"]]
    record_span_outputs(
        score_count=len(state["company_scores"]),
        company_scores=scores_payload,
    )
    record_json_artifact("workflow/company-scores.json", scores_payload)
    return state


def prepare_final_results_action(state: CompanyFitState) -> CompanyFitState:
    """Prepare final company fit results for export."""

    companies_by_name = {
        company.name: company
        for company in state.get("companies", [])
    }
    scores = sorted(
        state.get("company_scores", []),
        key=lambda score: score.overall_score,
        reverse=True,
    )
    result_count_maximum = get_result_count_maximum(state["company_search_criteria"])
    if result_count_maximum is not None:
        scores = scores[:result_count_maximum]

    final_results: list[FinalCompanyResult] = []
    for score in scores:
        company = companies_by_name.get(score.company_name)
        final_results.append(
            FinalCompanyResult(
                company_name=score.company_name,
                website_or_linkedin=(
                    company.website_or_linkedin if company is not None else ""
                ),
                location=company.location if company is not None else "",
                industry=company.industry if company is not None else "",
                company_size=company.company_size if company is not None else "",
                discovery_reason=(
                    company.discovery_reason if company is not None else ""
                ),
                overall_score=score.overall_score,
                axis_scores=[
                    FinalAxisResult(axis_name=axis.axis, score=axis.percentage)
                    for axis in score.axis_scores
                ],
            )
        )
    state["final_results"] = final_results
    mark_completed(state)
    return state

def _request_user_input_rephrase_after_guardrail(
    state: CompanyFitState,
) -> CompanyFitState:
    """Move the workflow into clarification mode after a recoverable guardrail failure."""

    if user_input_clarification_limit_reached(state):
        mark_failed(state, guardrail_rephrase_exhausted_message())
        return state

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

    record_span_outputs(
        guardrail_rephrase_requested=True,
        clarification_message=state.get("pending_clarification_message"),
        valid=False,
    )
    record_clarification_question(message, "user_input_interpretation")
    if blocked_message:
        record_guardrail_blocked_message(blocked_message, blocked_source)
    return state
