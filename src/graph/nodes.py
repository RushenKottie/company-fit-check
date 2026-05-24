"""LangGraph nodes for the current workflow."""

from time import perf_counter
from typing import Any

from mlflow.entities.span import SpanType

from llm.client import (
    build_cv_cleanup_due_to_guardrail_message,
    build_generic_llm_failure_message,
    build_rephrase_due_to_guardrail_message,
    build_rephrase_retry_exhausted_message,
    build_restart_due_to_guardrail_message,
    is_guardrail_4xx_error,
)
from logging_utils import get_logger
from models.state import CompanyFitState, UserInputInterpretation
from services.company_discovery import (
    build_zero_company_clarification_message,
    discover_companies,
    refine_company_search_criteria,
)
from services.company_scoring import score_companies
from services.cv_simplification import simplify_masked_cv
from services.user_input_interpretation_validation import (
    validate_user_input_interpretation,
)
from services.mlflow_tracking import (
    log_guardrail_blocked_user_message,
    log_clarification_question,
    log_json_artifact,
    log_text_artifact,
    traced_operation,
)
from services.pdf_text import extract_text_from_pdf_bytes
from services.pii_masking import mask_pii_locally
from services.user_input_interpretation import interpret_user_input

MAX_CLARIFICATION_ITERATIONS = 5
logger = get_logger(__name__)


def validate_search_prerequisites(state: CompanyFitState) -> str | None:
    """Return one deterministic error when company search prerequisites are missing."""

    if state.get("session_status") == "failed":
        return "Workflow is already in a failed state."
    if not (state.get("simplified_cv_text") or "").strip():
        return "Company search requires simplified CV text."
    if not state.get("axes"):
        return "Company search requires at least one matching axis."
    if state.get("company_search_criteria") is None:
        return "Company search requires initialized search criteria."
    return None


def validate_company_score_payload(state: CompanyFitState) -> str | None:
    """Return one deterministic error when the company score payload is malformed."""

    axes = state.get("axes", [])
    companies = state.get("companies", [])
    company_scores = state.get("company_scores", [])
    axis_names = [axis.name for axis in axes if axis.name.strip()]
    known_company_names = {company.name for company in companies}

    if not company_scores:
        return "Company scoring returned no company scores."

    for company_score in company_scores:
        if company_score.company_name not in known_company_names:
            return f"Company score references unknown company: {company_score.company_name}"

        score_by_axis = {
            axis_score.axis: axis_score.percentage
            for axis_score in company_score.axis_scores
        }
        if len(score_by_axis) != len(axis_names):
            return (
                f"Company {company_score.company_name} does not contain exactly one "
                "score for every axis."
            )

        for axis_name in axis_names:
            if axis_name not in score_by_axis:
                return f"Company {company_score.company_name} is missing axis score: {axis_name}"
            if not 0.0 <= score_by_axis[axis_name] <= 100.0:
                return (
                    f"Company {company_score.company_name} has out-of-range axis score: "
                    f"{axis_name}={score_by_axis[axis_name]}"
                )

        if not 0.0 <= company_score.overall_score <= 100.0:
            return (
                f"Company {company_score.company_name} has out-of-range overall score: "
                f"{company_score.overall_score}"
            )

    scored_company_names = {company_score.company_name for company_score in company_scores}
    if known_company_names != scored_company_names:
        missing = sorted(known_company_names - scored_company_names)
        return f"Missing company scores for: {', '.join(missing)}"

    return None


def build_case_result_summary(state: CompanyFitState) -> dict[str, Any]:
    """Return a compact JSON-safe state summary for deterministic eval artifacts."""

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
        "company_search_clarification_iterations": state.get(
            "company_search_clarification_iterations",
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
        "company_scores": [score.model_dump() for score in state.get("company_scores", [])],
        "pending_clarification_message": state.get("pending_clarification_message"),
        "latest_user_message_text": state.get("latest_user_message_text"),
        "latest_user_message_kind": state.get("latest_user_message_kind"),
        "guardrail_rephrase_source": state.get("guardrail_rephrase_source"),
    }


def extract_and_mask_cv_node(state: CompanyFitState) -> CompanyFitState:
    start = perf_counter()
    logger.info("Node start: extract_and_mask_cv")
    with traced_operation(
        "node.extract_and_mask_cv",
        span_type=SpanType.TASK,
        inputs={"session_status": state.get("session_status")},
    ) as span:
        if state.get("session_status") == "failed":
            logger.info("Node skip: extract_and_mask_cv because status=failed")
            _set_span_outputs(span, state)
            return state

        try:
            extracted_cv_text = extract_text_from_pdf_bytes(state["input"].cv_pdf_bytes)
        except Exception as exc:
            _fail(state, f"PDF text extraction failed: {exc}")
            _set_span_outputs(span, state)
            return state

        try:
            result = mask_pii_locally(extracted_cv_text)
        except Exception as exc:
            _fail(state, f"PII masking failed: {exc}")
            state["pii_masking_status"] = "failed"
            _set_span_outputs(span, state)
            return state

        state["masked_cv_text"] = result.text
        state["pii_masking_status"] = "passed"
        log_text_artifact("workflow/masked-cv.txt", result.text)
        _set_span_outputs(
            span,
            state,
            masked_cv_chars=len(result.text),
            pii_masking_status=state.get("pii_masking_status"),
        )
    logger.info(
        "Node end: extract_and_mask_cv masked_chars=%s duration_ms=%.1f",
        len(result.text),
        (perf_counter() - start) * 1000,
    )
    return state


def validate_pii_masking_node(state: CompanyFitState) -> CompanyFitState:
    start = perf_counter()
    logger.info("Node start: validate_pii_masking")
    with traced_operation(
        "node.validate_pii_masking",
        span_type=SpanType.GUARDRAIL,
        inputs={"pii_masking_status": state.get("pii_masking_status")},
    ) as span:
        if state.get("session_status") == "failed":
            logger.info("Node skip: validate_pii_masking because status=failed")
            _set_span_outputs(span, state)
            return state

        masked = state.get("masked_cv_text") or ""
        if state.get("pii_masking_status") != "passed":
            _fail(state, "PII masking did not complete successfully.")
        elif not masked:
            _fail(state, "PII masking produced empty output.")
        _set_span_outputs(span, state, masked_cv_chars=len(masked))
    logger.info(
        "Node end: validate_pii_masking status=%s masked_chars=%s duration_ms=%.1f",
        state.get("session_status"),
        len(masked),
        (perf_counter() - start) * 1000,
    )
    return state


def simplify_cv_node(state: CompanyFitState) -> CompanyFitState:
    start = perf_counter()
    logger.info("Node start: simplify_cv")
    with traced_operation(
        "node.simplify_cv",
        span_type=SpanType.CHAIN,
        inputs={"masked_cv_chars": len(state.get("masked_cv_text") or "")},
    ) as span:
        if state.get("session_status") == "failed":
            logger.info("Node skip: simplify_cv because status=failed")
            _set_span_outputs(span, state)
            return state

        try:
            state["simplified_cv_text"] = simplify_masked_cv(state["masked_cv_text"] or "")
        except Exception as exc:
            logger.exception("CV simplification failed.")
            if is_guardrail_4xx_error(exc):
                _fail(state, build_cv_cleanup_due_to_guardrail_message())
            else:
                _fail(state, build_generic_llm_failure_message())
        if state.get("simplified_cv_text"):
            log_text_artifact(
                "workflow/simplified-cv.txt",
                state["simplified_cv_text"],
            )
        _set_span_outputs(span, state, simplified_cv_chars=len(state.get("simplified_cv_text") or ""))
    logger.info(
        "Node end: simplify_cv status=%s simplified_chars=%s duration_ms=%.1f",
        state.get("session_status"),
        len(state.get("simplified_cv_text") or ""),
        (perf_counter() - start) * 1000,
    )
    return state


def interpret_user_input_node(state: CompanyFitState) -> CompanyFitState:
    start = perf_counter()
    logger.info("Node start: interpret_user_input")
    with traced_operation(
        "node.interpret_user_input",
        span_type=SpanType.CHAIN,
        inputs={
            "prompt": state["input"].prompt,
            "clarification_present": bool(state.get("latest_clarification_response")),
            "previous_axes": [axis.model_dump() for axis in state.get("axes", [])],
        },
    ) as span:
        if state.get("session_status") == "failed":
            logger.info("Node skip: interpret_user_input because status=failed")
            _set_span_outputs(span, state)
            return state

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
                if _request_user_input_rephrase_after_guardrail(state):
                    _set_span_outputs(
                        span,
                        state,
                        guardrail_rephrase_requested=True,
                        clarification_message=state.get("pending_clarification_message"),
                    )
                    return state
                _set_span_outputs(span, state)
                return state
            _fail(state, build_generic_llm_failure_message())
            _set_span_outputs(span, state)
            return state

        state["company_search_criteria"] = user_input_interpretation.company_search_criteria
        state["axes"] = user_input_interpretation.axes
        log_json_artifact(
            f"workflow/user-input-interpretation-{_artifact_timestamp()}.json",
            {
                "company_search_criteria": user_input_interpretation.company_search_criteria.model_dump(),
                "axes": [axis.model_dump() for axis in user_input_interpretation.axes],
            },
        )
        if user_input_interpretation != previous_user_input_interpretation:
            state["user_input_interpretation_clarification_iterations"] = 0
        _set_span_outputs(
            span,
            state,
            axes=[axis.model_dump() for axis in user_input_interpretation.axes],
            company_search_criteria=user_input_interpretation.company_search_criteria.model_dump(),
        )
    logger.info(
        "Node end: interpret_user_input axes=%s criteria_fields=%s duration_ms=%.1f",
        len(state.get("axes", [])),
        sum(
            1
            for value in state["company_search_criteria"].model_dump().values()
            if value
        ),
        (perf_counter() - start) * 1000,
    )
    return state


def validate_user_input_interpretation_node(state: CompanyFitState) -> CompanyFitState:
    start = perf_counter()
    logger.info("Node start: validate_user_input_interpretation")
    with traced_operation(
        "node.validate_user_input_interpretation",
        span_type=SpanType.GUARDRAIL,
        inputs={
            "axes": [axis.model_dump() for axis in state.get("axes", [])],
            "clarification_present": bool(state.get("latest_clarification_response")),
        },
    ) as span:
        if state.get("session_status") == "failed":
            logger.info("Node skip: validate_user_input_interpretation because status=failed")
            _set_span_outputs(span, state)
            return state

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
                if _request_user_input_rephrase_after_guardrail(state):
                    _set_span_outputs(
                        span,
                        state,
                        valid=False,
                        guardrail_rephrase_requested=True,
                        clarification_message=state.get("pending_clarification_message"),
                    )
                    return state
                _set_span_outputs(span, state)
                return state
            _fail(state, build_generic_llm_failure_message())
            _set_span_outputs(span, state)
            return state
        if valid:
            state["pending_clarification_message"] = None
            state["clarification_target"] = None
            state["guardrail_rephrase_source"] = None
            state["user_input_interpretation_clarification_iterations"] = 0
            state["session_status"] = "running"
            state["latest_clarification_response"] = None
        else:
            if (
                state.get("user_input_interpretation_clarification_iterations", 0)
                >= MAX_CLARIFICATION_ITERATIONS
            ):
                _fail(
                    state,
                    "User-input interpretation clarification did not make progress after 5 attempts.",
                )
                _set_span_outputs(span, state, valid=valid, clarification_message=message)
                return state
            state["pending_clarification_message"] = message
            state["clarification_target"] = "user_input_interpretation"
            state["session_status"] = "needs_clarification"
            if message:
                log_clarification_question(message, target="user_input_interpretation")
        _set_span_outputs(span, state, valid=valid, clarification_message=message)
    logger.info(
        "Node end: validate_user_input_interpretation valid=%s status=%s duration_ms=%.1f",
        valid,
        state.get("session_status"),
        (perf_counter() - start) * 1000,
    )
    return state


def search_companies_node(state: CompanyFitState) -> CompanyFitState:
    start = perf_counter()
    logger.info("Node start: search_companies")
    with traced_operation(
        "node.search_companies",
        span_type=SpanType.RETRIEVER,
        inputs={
            "company_search_criteria": (
                state["company_search_criteria"].model_dump()
                if state.get("company_search_criteria") is not None
                else None
            )
        },
    ) as span:
        if state.get("session_status") == "failed":
            logger.info("Node skip: search_companies because status=failed")
            _set_span_outputs(span, state)
            return state

        prerequisite_error = validate_search_prerequisites(state)
        if prerequisite_error:
            _fail(state, prerequisite_error)
            _set_span_outputs(span, state, prerequisite_error=prerequisite_error)
            return state

        try:
            state["companies"] = discover_companies(
                company_search_criteria=state["company_search_criteria"],
            )
        except Exception as exc:
            logger.exception("Company discovery failed.")
            if is_guardrail_4xx_error(exc):
                _fail(state, build_restart_due_to_guardrail_message("company search"))
            else:
                _fail(state, build_generic_llm_failure_message())
            _set_span_outputs(span, state)
            return state

        if not state["companies"]:
            if (
                state.get("company_search_clarification_iterations", 0)
                >= MAX_CLARIFICATION_ITERATIONS
            ):
                _fail(
                    state,
                    "Company-search clarification did not make progress after 5 attempts.",
                )
                _set_span_outputs(span, state, company_count=0)
                return state
            try:
                state["pending_clarification_message"] = (
                    build_zero_company_clarification_message(
                        state["company_search_criteria"]
                    )
                )
            except Exception as exc:
                logger.exception("Zero-result clarification failed.")
                if is_guardrail_4xx_error(exc):
                    _fail(state, build_restart_due_to_guardrail_message("company search"))
                else:
                    _fail(state, build_generic_llm_failure_message())
                _set_span_outputs(span, state, company_count=0)
                return state

            state["clarification_target"] = "company_search"
            state["session_status"] = "needs_clarification"
            if state.get("pending_clarification_message"):
                log_clarification_question(
                    state["pending_clarification_message"],
                    target="company_search",
                )
            _set_span_outputs(
                span,
                state,
                company_count=0,
                clarification_message=state.get("pending_clarification_message"),
            )
            logger.info(
                "Node end: search_companies status=%s companies=0 duration_ms=%.1f",
                state.get("session_status"),
                (perf_counter() - start) * 1000,
            )
            return state

        log_json_artifact(
            f"workflow/discovered-companies-{_artifact_timestamp()}.json",
            [company.model_dump() for company in state["companies"]],
        )
        state["pending_clarification_message"] = None
        state["clarification_target"] = None
        state["guardrail_rephrase_source"] = None
        state["company_search_clarification_iterations"] = 0
        state["session_status"] = "running"
        _set_span_outputs(
            span,
            state,
            company_count=len(state["companies"]),
            companies=[company.model_dump() for company in state["companies"]],
        )
    logger.info(
        "Node end: search_companies status=%s companies=%s duration_ms=%.1f",
        state.get("session_status"),
        len(state.get("companies", [])),
        (perf_counter() - start) * 1000,
    )
    return state


def score_companies_node(state: CompanyFitState) -> CompanyFitState:
    start = perf_counter()
    logger.info("Node start: score_companies")
    with traced_operation(
        "node.score_companies",
        span_type=SpanType.EVALUATOR,
        inputs={
            "company_count": len(state.get("companies", [])),
            "axes": [axis.model_dump() for axis in state.get("axes", [])],
        },
    ) as span:
        if state.get("session_status") == "failed":
            logger.info("Node skip: score_companies because status=failed")
            _set_span_outputs(span, state)
            return state

        try:
            state["company_scores"] = score_companies(
                companies=state.get("companies", []),
                simplified_cv_text=state.get("simplified_cv_text") or "",
                axes=state.get("axes", []),
            )
        except Exception as exc:
            logger.exception("Company scoring failed.")
            if is_guardrail_4xx_error(exc):
                _fail(state, build_restart_due_to_guardrail_message("company scoring"))
            else:
                _fail(state, build_generic_llm_failure_message())
            _set_span_outputs(span, state)
            return state

        payload_error = validate_company_score_payload(state)
        if payload_error:
            _fail(state, payload_error)
            _set_span_outputs(span, state, payload_error=payload_error)
            return state

        log_json_artifact(
            "workflow/company-scores.json",
            [score.model_dump() for score in state["company_scores"]],
        )
        state["session_status"] = "completed"
        _set_span_outputs(
            span,
            state,
            score_count=len(state["company_scores"]),
            company_scores=[score.model_dump() for score in state["company_scores"]],
        )
    logger.info(
        "Node end: score_companies scores=%s duration_ms=%.1f",
        len(state.get("company_scores", [])),
        (perf_counter() - start) * 1000,
    )
    return state


def refine_company_search_node(state: CompanyFitState) -> CompanyFitState:
    start = perf_counter()
    logger.info("Node start: refine_company_search")
    with traced_operation(
        "node.refine_company_search",
        span_type=SpanType.CHAIN,
        inputs={
            "company_search_criteria": state["company_search_criteria"].model_dump(),
            "clarification": state.get("latest_clarification_response"),
        },
    ) as span:
        if state.get("session_status") == "failed":
            logger.info("Node skip: refine_company_search because status=failed")
            _set_span_outputs(span, state)
            return state

        clarification = state.get("latest_clarification_response")
        if not clarification:
            _fail(state, "Missing company-search clarification response.")
            _set_span_outputs(span, state)
            return state

        try:
            previous_criteria = state["company_search_criteria"]
            state["company_search_criteria"] = refine_company_search_criteria(
                company_search_criteria=state["company_search_criteria"],
                user_clarification=clarification,
            )
        except Exception as exc:
            logger.exception("Company-search refinement failed.")
            if is_guardrail_4xx_error(exc):
                _fail(state, build_restart_due_to_guardrail_message("company search refinement"))
            else:
                _fail(state, build_generic_llm_failure_message())
            _set_span_outputs(span, state)
            return state

        log_json_artifact(
            f"workflow/refined-company-search-{_artifact_timestamp()}.json",
            state["company_search_criteria"].model_dump(),
        )
        if state["company_search_criteria"] != previous_criteria:
            state["company_search_clarification_iterations"] = 0
        state["latest_clarification_response"] = None
        state["pending_clarification_message"] = None
        state["clarification_target"] = None
        state["guardrail_rephrase_source"] = None
        state["session_status"] = "running"
        _set_span_outputs(
            span,
            state,
            company_search_criteria=state["company_search_criteria"].model_dump(),
        )
    logger.info(
        "Node end: refine_company_search duration_ms=%.1f",
        (perf_counter() - start) * 1000,
    )
    return state


def _fail(state: CompanyFitState, message: str) -> None:
    logger.error("Workflow state marked failed: %s", message)
    state["session_status"] = "failed"
    state["error"] = message


def _request_user_input_rephrase_after_guardrail(state: CompanyFitState) -> bool:
    """Move the workflow into clarification mode after a recoverable guardrail failure."""

    if (
        state.get("user_input_interpretation_clarification_iterations", 0)
        >= MAX_CLARIFICATION_ITERATIONS
    ):
        _fail(state, build_rephrase_retry_exhausted_message())
        return False

    blocked_source = _guardrail_blocked_message_source(state)
    blocked_message = _guardrail_blocked_message_text(state)
    if blocked_message:
        log_guardrail_blocked_user_message(
            run_id=state.get("run_id"),
            message=blocked_message,
            source=blocked_source,
        )

    message = build_rephrase_due_to_guardrail_message()
    state["pending_clarification_message"] = message
    state["clarification_target"] = "user_input_interpretation"
    state["session_status"] = "needs_clarification"
    state["error"] = None
    state["guardrail_rephrase_source"] = blocked_source
    if blocked_source == "prompt":
        state["input"].prompt = ""
    else:
        state["latest_clarification_response"] = None
    state["latest_user_message_text"] = None
    state["latest_user_message_kind"] = None
    state["user_input_interpretation_clarification_iterations"] = (
        state.get("user_input_interpretation_clarification_iterations", 0) + 1
    )
    log_clarification_question(message, target="user_input_interpretation")
    return True


def _guardrail_blocked_message_source(state: CompanyFitState) -> str:
    """Return whether the blocked user message came from the prompt or clarification."""

    kind = state.get("latest_user_message_kind")
    if kind in {"prompt", "clarification"}:
        return kind
    return "clarification" if state.get("latest_clarification_response") else "prompt"


def _guardrail_blocked_message_text(state: CompanyFitState) -> str:
    """Return the raw user message that most likely triggered the guardrail."""

    latest = (state.get("latest_user_message_text") or "").strip()
    if latest:
        return latest
    if state.get("latest_clarification_response"):
        return str(state.get("latest_clarification_response")).strip()
    if state.get("input") is not None:
        return state["input"].prompt.strip()
    return ""


def _artifact_timestamp() -> str:
    """Return a compact timestamp for one artifact filename."""

    return str(int(perf_counter() * 1000))


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
