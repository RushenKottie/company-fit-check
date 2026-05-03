"""LangGraph nodes for the current workflow."""

from time import perf_counter

from mlflow.entities.span import SpanType

from company_fit_check.logging_utils import get_logger
from company_fit_check.models.state import CompanyFitState, PromptInterpretation
from company_fit_check.services.company_discovery import (
    build_zero_company_clarification_message,
    discover_companies,
    refine_company_search_criteria,
)
from company_fit_check.services.company_scoring import score_companies
from company_fit_check.services.cv_simplification import simplify_masked_cv
from company_fit_check.services.interpretation_validation import (
    validate_interpretation,
)
from company_fit_check.services.mlflow_tracking import (
    log_clarification_question,
    log_json_artifact,
    log_text_artifact,
    traced_operation,
)
from company_fit_check.services.pdf_text import extract_text_from_pdf_bytes
from company_fit_check.services.pii_masking import mask_pii_locally
from company_fit_check.services.prompt_interpretation import interpret_prompt

MAX_CLARIFICATION_ITERATIONS = 5
logger = get_logger(__name__)


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
            _fail(state, f"CV simplification failed: {exc}")
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


def interpret_prompt_node(state: CompanyFitState) -> CompanyFitState:
    start = perf_counter()
    logger.info("Node start: interpret_prompt")
    with traced_operation(
        "node.interpret_prompt",
        span_type=SpanType.CHAIN,
        inputs={
            "prompt": state["input"].prompt,
            "clarification_present": bool(state.get("latest_clarification_response")),
            "previous_axes": [axis.model_dump() for axis in state.get("axes", [])],
        },
    ) as span:
        if state.get("session_status") == "failed":
            logger.info("Node skip: interpret_prompt because status=failed")
            _set_span_outputs(span, state)
            return state

        previous_interpretation = PromptInterpretation(
            company_search_criteria=state["company_search_criteria"],
            axes=state.get("axes", []),
        )
        try:
            interpretation = interpret_prompt(
                prompt=state["input"].prompt,
                simplified_cv_text=state["simplified_cv_text"] or "",
                clarification=state.get("latest_clarification_response"),
                previous_axes=state.get("axes", []),
            )
        except Exception as exc:
            _fail(state, f"Prompt interpretation failed: {exc}")
            _set_span_outputs(span, state)
            return state

        state["company_search_criteria"] = interpretation.company_search_criteria
        state["axes"] = interpretation.axes
        log_json_artifact(
            f"workflow/interpretation-{_artifact_timestamp()}.json",
            {
                "company_search_criteria": interpretation.company_search_criteria.model_dump(),
                "axes": [axis.model_dump() for axis in interpretation.axes],
            },
        )
        if interpretation != previous_interpretation:
            state["interpretation_clarification_iterations"] = 0
        _set_span_outputs(
            span,
            state,
            axes=[axis.model_dump() for axis in interpretation.axes],
            company_search_criteria=interpretation.company_search_criteria.model_dump(),
        )
    logger.info(
        "Node end: interpret_prompt axes=%s criteria_fields=%s duration_ms=%.1f",
        len(state.get("axes", [])),
        sum(
            1
            for value in state["company_search_criteria"].model_dump().values()
            if value
        ),
        (perf_counter() - start) * 1000,
    )
    return state


def validate_interpretation_node(state: CompanyFitState) -> CompanyFitState:
    start = perf_counter()
    logger.info("Node start: validate_interpretation")
    with traced_operation(
        "node.validate_interpretation",
        span_type=SpanType.GUARDRAIL,
        inputs={
            "axes": [axis.model_dump() for axis in state.get("axes", [])],
            "clarification_present": bool(state.get("latest_clarification_response")),
        },
    ) as span:
        if state.get("session_status") == "failed":
            logger.info("Node skip: validate_interpretation because status=failed")
            _set_span_outputs(span, state)
            return state

        valid, message = validate_interpretation(
            axes=state.get("axes", []),
            simplified_cv_text=state.get("simplified_cv_text") or "",
            prompt=state["input"].prompt,
            clarification=state.get("latest_clarification_response"),
        )
        if valid:
            state["pending_clarification_message"] = None
            state["clarification_target"] = None
            state["interpretation_clarification_iterations"] = 0
            state["session_status"] = "running"
            state["latest_clarification_response"] = None
        else:
            if (
                state.get("interpretation_clarification_iterations", 0)
                >= MAX_CLARIFICATION_ITERATIONS
            ):
                _fail(
                    state,
                    "Interpretation clarification did not make progress after 5 attempts.",
                )
                _set_span_outputs(span, state, valid=valid, clarification_message=message)
                return state
            state["pending_clarification_message"] = message
            state["clarification_target"] = "interpretation"
            state["session_status"] = "needs_clarification"
            if message:
                log_clarification_question(message, target="interpretation")
        _set_span_outputs(span, state, valid=valid, clarification_message=message)
    logger.info(
        "Node end: validate_interpretation valid=%s status=%s duration_ms=%.1f",
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
        inputs={"company_search_criteria": state["company_search_criteria"].model_dump()},
    ) as span:
        if state.get("session_status") == "failed":
            logger.info("Node skip: search_companies because status=failed")
            _set_span_outputs(span, state)
            return state

        try:
            state["companies"] = discover_companies(
                company_search_criteria=state["company_search_criteria"],
            )
        except Exception as exc:
            _fail(state, f"Company discovery failed: {exc}")
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
                _fail(state, f"Zero-result clarification failed: {exc}")
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
            _fail(state, f"Company scoring failed: {exc}")
            _set_span_outputs(span, state)
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
            _fail(state, f"Company-search refinement failed: {exc}")
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
