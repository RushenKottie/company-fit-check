"""Shared deterministic checks used by pytest and MLflow scorers."""

from __future__ import annotations

import csv
import io
import re

from application.workflow_policy import MAX_CLARIFICATION_ITERATIONS
from evals.deterministic.models import CaseExecutionResult, CheckResult
from graph.node_names import WorkflowNodeName
from llm.client import (
    build_cv_cleanup_due_to_guardrail_message,
    build_generic_llm_failure_message,
    build_rephrase_due_to_guardrail_message,
)
from application.result_exports import RESULT_CSV_BASE_COLUMNS

EMAIL_PATTERN = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", flags=re.IGNORECASE)
PHONE_PATTERN = re.compile(r"\+?\d[\d\-\s()]{7,}\d")
PERSONAL_LINK_PATTERNS = [
    re.compile(r"https?://(?:www\.)?linkedin\.com/in/[^\s)]+", flags=re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?github\.com/[^\s/]+/?", flags=re.IGNORECASE),
    re.compile(r"mailto:[^\s]+", flags=re.IGNORECASE),
]
ADDRESS_LABELS = {"address", "home address", "residential address"}
BIRTH_DATE_LABELS = {"date of birth", "birth date", "dob"}
CSV_AXIS_COLUMN_PATTERN = re.compile(r"^axis_\d+_.+_score$")


def check_llm_uses_masked_cv_not_raw_cv(
    result: CaseExecutionResult,
    case_inputs: dict,
) -> CheckResult:
    """Verify every LLM prompt that includes CV text uses the masked CV, not the raw CV."""

    spans = result.observed_spans
    raw_cv_text = case_inputs.get("raw_cv_text", "")
    masked_cv_text = result.final_state_summary.get("masked_cv_text") or ""
    offending_spans: list[str] = []
    cv_llm_spans: list[str] = []

    for span in spans:
        span_name = str(span.get("name", ""))
        if not span_name.startswith("llm."):
            continue
        prompt_text = "\n".join(
            message.get("content", "")
            for message in span.get("prompt_messages", [])
        )
        raw_cv_reached_llm = bool(raw_cv_text) and raw_cv_text in prompt_text
        masked_cv_reached_llm = bool(masked_cv_text) and masked_cv_text in prompt_text
        if raw_cv_reached_llm or masked_cv_reached_llm:
            cv_llm_spans.append(span_name)
        if raw_cv_reached_llm:
            offending_spans.append(span_name)

    passed = bool(cv_llm_spans) and not offending_spans
    return CheckResult(
        name="llm_uses_masked_cv_not_raw_cv",
        passed=passed,
        reason=(
            "Every LLM prompt that included CV content used the masked CV text and never the raw CV text."
            if passed
            else "No masked-CV LLM prompt was observed or at least one LLM prompt included raw CV text."
        ),
        details={
            "cv_llm_spans": cv_llm_spans,
            "offending_spans": offending_spans,
        },
    )


def check_masked_cv_has_no_contact_details(
    result: CaseExecutionResult,
    case_inputs: dict,
) -> CheckResult:
    """Verify the masked CV is populated and free of contact or identifying details."""

    masked_cv_text = result.final_state_summary.get("masked_cv_text") or ""
    matches = scan_text_for_pii(masked_cv_text)
    return CheckResult(
        name="masked_cv_has_no_contact_details",
        passed=bool(masked_cv_text.strip()) and not matches,
        reason=(
            "Masked CV is non-empty and contains no contact or identifying details."
            if masked_cv_text.strip() and not matches
            else "Masked CV is empty or still contains contact or identifying details."
        ),
        details={"matches": matches},
    )


def check_workflow_stops_on_masking_failure(
    result: CaseExecutionResult,
    case_inputs: dict,
) -> CheckResult:
    """Verify masking failure stops the workflow before downstream stages run."""

    downstream_spans = {
        WorkflowNodeName.SIMPLIFY_CV.span_name,
        WorkflowNodeName.INTERPRET_USER_INPUT.span_name,
        WorkflowNodeName.SEARCH_COMPANIES.span_name,
        WorkflowNodeName.SCORE_COMPANIES.span_name,
    }
    unexpected = [name for name in result.executed_span_names if name in downstream_spans]
    passed = result.status == "failed" and not unexpected
    return CheckResult(
        name="workflow_stops_on_masking_failure",
        passed=passed,
        reason="Masking failure stopped the workflow cleanly." if passed else "Downstream stages ran after masking failure or the state did not fail.",
        details={"unexpected_spans": unexpected},
    )


def check_company_search_blocked_until_prereqs_valid(
    result: CaseExecutionResult,
    case_inputs: dict,
) -> CheckResult:
    """Verify company search does not run when prerequisites are missing."""

    discovery_spans = [name for name in result.executed_span_names if name == "llm.discover_companies"]
    error = result.error or ""
    passed = result.status == "failed" and "Company search requires" in error and not discovery_spans
    return CheckResult(
        name="company_search_blocked_until_prereqs_valid",
        passed=passed,
        reason="Company search was blocked before discovery executed." if passed else "Company search prerequisites were not enforced deterministically.",
        details={"error": error, "discovery_spans": discovery_spans},
    )


def check_clarification_loop_is_bounded(
    result: CaseExecutionResult,
    case_inputs: dict,
) -> CheckResult:
    """Verify the clarification loop terminates at the configured max."""

    error = result.error or ""
    expected_fragment = (
        f"did not make progress after {MAX_CLARIFICATION_ITERATIONS} attempts"
    )
    passed = result.status == "failed" and expected_fragment in error
    return CheckResult(
        name="clarification_loop_is_bounded",
        passed=passed,
        reason="Clarification loop is bounded." if passed else "Clarification loop was not bounded as expected.",
        details={
            "error": error,
            "expected_fragment": expected_fragment,
            "max_clarification_iterations": MAX_CLARIFICATION_ITERATIONS,
        },
    )


def check_guardrail_4xx_requests_rephrase(
    result: CaseExecutionResult,
    case_inputs: dict,
) -> CheckResult:
    """Verify guardrail 4xx failures ask the user to reformulate without failing."""

    passed = (
        result.status == "needs_clarification"
        and result.uncaught_exception is None
        and result.error is None
        and result.final_state_summary.get("clarification_target") == "user_input_interpretation"
        and result.final_state_summary.get("pending_clarification_message")
        == build_rephrase_due_to_guardrail_message()
        and result.final_state_summary.get("guardrail_rephrase_source") == "prompt"
        and result.final_state_summary.get("latest_user_message_text") is None
        and result.final_state_summary.get("latest_user_message_kind") is None
        and result.final_state_summary.get("user_input_interpretation_clarification_iterations", 0)
        > 0
    )
    return CheckResult(
        name="guardrail_4xx_requests_rephrase",
        passed=passed,
        reason=(
            "Guardrail 4xx requested reformulation without failing the workflow."
            if passed
            else "Guardrail 4xx did not route into the expected reformulation flow."
        ),
        details={
            "status": result.status,
            "error": result.error,
            "clarification_target": result.final_state_summary.get("clarification_target"),
            "pending_clarification_message": result.final_state_summary.get("pending_clarification_message"),
            "guardrail_rephrase_source": result.final_state_summary.get("guardrail_rephrase_source"),
            "latest_user_message_text": result.final_state_summary.get("latest_user_message_text"),
            "latest_user_message_kind": result.final_state_summary.get("latest_user_message_kind"),
            "iterations": result.final_state_summary.get(
                "user_input_interpretation_clarification_iterations",
            ),
            "uncaught_exception": result.uncaught_exception,
        },
    )


def check_guardrail_4xx_rephrase_stops_workflow(
    result: CaseExecutionResult,
    case_inputs: dict,
) -> CheckResult:
    """Verify workflow-level guardrail rephrase stops before downstream stages run."""

    blocked_spans = {
        WorkflowNodeName.VALIDATE_USER_INPUT_INTERPRETATION.span_name,
        WorkflowNodeName.SEARCH_COMPANIES.span_name,
        WorkflowNodeName.SCORE_COMPANIES.span_name,
    }
    unexpected = [name for name in result.executed_span_names if name in blocked_spans]
    passed = (
        result.status == "needs_clarification"
        and result.error is None
        and result.final_state_summary.get("clarification_target") == "user_input_interpretation"
        and not unexpected
    )
    return CheckResult(
        name="guardrail_4xx_rephrase_stops_workflow",
        passed=passed,
        reason=(
            "Workflow stopped immediately after requesting reformulation."
            if passed
            else "Workflow continued into downstream stages after requesting reformulation."
        ),
        details={
            "unexpected_spans": unexpected,
            "executed_span_names": result.executed_span_names,
            "status": result.status,
            "error": result.error,
        },
    )


def check_prompt_rephrase_replaces_blocked_prompt(
    result: CaseExecutionResult,
    case_inputs: dict,
) -> CheckResult:
    """Verify a blocked prompt can be replaced by a reformulated prompt on resume."""

    final_prompt = (
        (result.final_state_summary.get("input") or {}).get("prompt", "")
        if isinstance(result.final_state_summary.get("input"), dict)
        else ""
    )
    original_prompt = case_inputs.get("prompt", "")
    passed = (
        result.status == "completed"
        and result.error is None
        and result.final_state_summary.get("guardrail_rephrase_source") is None
        and result.final_state_summary.get("latest_user_message_kind") == "prompt"
        and "safe reformulation" in final_prompt.lower()
        and original_prompt not in final_prompt
    )
    return CheckResult(
        name="prompt_rephrase_replaces_blocked_prompt",
        passed=passed,
        reason=(
            "Blocked prompt was replaced by the reformulated prompt and the workflow completed."
            if passed
            else "Workflow did not complete after replacing the blocked prompt."
        ),
        details={
            "status": result.status,
            "error": result.error,
            "guardrail_rephrase_source": result.final_state_summary.get("guardrail_rephrase_source"),
            "latest_user_message_kind": result.final_state_summary.get("latest_user_message_kind"),
            "final_prompt": final_prompt,
        },
    )


def check_guardrail_4xx_fails_with_restart(
    result: CaseExecutionResult,
    case_inputs: dict,
) -> CheckResult:
    """Verify guardrail 4xx failures ask the user to restart the session."""

    error = result.error or ""
    passed = (
        result.status == "failed"
        and result.uncaught_exception is None
        and error.startswith("Sorry, I couldn't continue because the model blocked this request")
        and "Please start a new session with another prompt or CV." in error
        and result.final_state_summary.get("clarification_target") is None
    )
    return CheckResult(
        name="guardrail_4xx_fails_with_restart",
        passed=passed,
        reason=(
            "Guardrail 4xx produced the expected restart message."
            if passed
            else "Guardrail 4xx did not produce the expected restart behavior."
        ),
        details={
            "error": error,
            "clarification_target": result.final_state_summary.get("clarification_target"),
            "uncaught_exception": result.uncaught_exception,
        },
    )


def check_guardrail_4xx_fails_with_cv_cleanup(
    result: CaseExecutionResult,
    case_inputs: dict,
) -> CheckResult:
    """Verify CV simplification guardrail 4xx asks the user to clean the CV."""

    error = result.error or ""
    passed = (
        result.status == "failed"
        and result.uncaught_exception is None
        and error == build_cv_cleanup_due_to_guardrail_message()
    )
    return CheckResult(
        name="guardrail_4xx_fails_with_cv_cleanup",
        passed=passed,
        reason=(
            "Guardrail 4xx produced the expected CV cleanup message."
            if passed
            else "Guardrail 4xx did not produce the expected CV cleanup message."
        ),
        details={"error": error, "uncaught_exception": result.uncaught_exception},
    )


def check_generic_llm_failure_is_sanitized(
    result: CaseExecutionResult,
    case_inputs: dict,
) -> CheckResult:
    """Verify generic LLM failures show the sanitized support message."""

    error = result.error or ""
    passed = (
        result.status == "failed"
        and result.uncaught_exception is None
        and error == build_generic_llm_failure_message()
        and "Injected deterministic failure" not in error
    )
    return CheckResult(
        name="generic_llm_failure_is_sanitized",
        passed=passed,
        reason=(
            "Generic LLM failure was sanitized for the user."
            if passed
            else "Generic LLM failure leaked internal exception text or used the wrong message."
        ),
        details={"error": error, "uncaught_exception": result.uncaught_exception},
    )


def check_score_shape_is_valid_for_all_companies(
    result: CaseExecutionResult,
    case_inputs: dict,
) -> CheckResult:
    """Verify score validation helper accepted the payload."""

    passed = result.helper_output in {None, ""}
    return CheckResult(
        name="score_shape_is_valid_for_all_companies",
        passed=passed,
        reason="Score payload is valid." if passed else "Score payload validator returned an error.",
        details={"helper_output": result.helper_output},
    )


def check_csv_schema_is_valid(
    result: CaseExecutionResult,
    case_inputs: dict,
) -> CheckResult:
    """Verify CSV output exists and follows the expected schema."""

    csv_artifact = result.csv_artifact or {}
    content = csv_artifact.get("content", "")
    if not content:
        return CheckResult(
            name="csv_schema_is_valid",
            passed=False,
            reason="CSV artifact was not generated.",
        )

    rows = list(csv.DictReader(io.StringIO(content)))
    fieldnames = list(rows[0].keys()) if rows else csv.DictReader(io.StringIO(content)).fieldnames or []
    base_columns = set(RESULT_CSV_BASE_COLUMNS)
    axis_columns = [name for name in fieldnames if name not in base_columns]
    companies = result.final_state_summary.get("company_scores", [])
    passed = (
        base_columns.issubset(set(fieldnames))
        and len(rows) == len(companies)
        and all(CSV_AXIS_COLUMN_PATTERN.match(column or "") for column in axis_columns)
    )
    return CheckResult(
        name="csv_schema_is_valid",
        passed=passed,
        reason="CSV schema and row count are valid." if passed else "CSV schema or row count is invalid.",
        details={"fieldnames": fieldnames, "row_count": len(rows)},
    )


CHECK_FUNCTIONS = {
    "llm_uses_masked_cv_not_raw_cv": check_llm_uses_masked_cv_not_raw_cv,
    "masked_cv_has_no_contact_details": check_masked_cv_has_no_contact_details,
    "workflow_stops_on_masking_failure": check_workflow_stops_on_masking_failure,
    "company_search_blocked_until_prereqs_valid": check_company_search_blocked_until_prereqs_valid,
    "clarification_loop_is_bounded": check_clarification_loop_is_bounded,
    "guardrail_4xx_requests_rephrase": check_guardrail_4xx_requests_rephrase,
    "guardrail_4xx_rephrase_stops_workflow": check_guardrail_4xx_rephrase_stops_workflow,
    "prompt_rephrase_replaces_blocked_prompt": check_prompt_rephrase_replaces_blocked_prompt,
    "guardrail_4xx_fails_with_restart": check_guardrail_4xx_fails_with_restart,
    "guardrail_4xx_fails_with_cv_cleanup": check_guardrail_4xx_fails_with_cv_cleanup,
    "generic_llm_failure_is_sanitized": check_generic_llm_failure_is_sanitized,
    "score_shape_is_valid_for_all_companies": check_score_shape_is_valid_for_all_companies,
    "csv_schema_is_valid": check_csv_schema_is_valid,
}


def run_requested_checks(result: CaseExecutionResult, check_names: list[str]) -> list[CheckResult]:
    """Run the requested deterministic checks for one result."""

    if result.uncaught_exception:
        return [
            CheckResult(
                name=check_name,
                passed=False,
                reason=result.uncaught_exception,
                details={"error": result.error},
            )
            for check_name in check_names
        ]

    outputs: list[CheckResult] = []
    for check_name in check_names:
        outputs.append(CHECK_FUNCTIONS[check_name](result, result.case_inputs))
    return outputs


def scan_text_for_pii(text: str) -> list[str]:
    """Return contact or identifying PII-like matches found in text."""

    matches: list[str] = []
    if not text:
        return matches
    matches.extend(match.group(0) for match in EMAIL_PATTERN.finditer(text))
    matches.extend(_phone_like_matches(text))
    for pattern in PERSONAL_LINK_PATTERNS:
        matches.extend(match.group(0) for match in pattern.finditer(text))
    matches.extend(_find_unmasked_labeled_lines(text, ADDRESS_LABELS, "<ADDRESS>"))
    matches.extend(
        _find_unmasked_labeled_lines(text, BIRTH_DATE_LABELS, "<DATE_OF_BIRTH>")
    )
    return matches


def _phone_like_matches(text: str) -> list[str]:
    """Return only phone-like matches with at least ten digits."""

    outputs: list[str] = []
    for match in PHONE_PATTERN.finditer(text):
        candidate = match.group(0)
        digit_count = sum(character.isdigit() for character in candidate)
        if digit_count >= 10:
            outputs.append(candidate)
    return outputs


def _find_unmasked_labeled_lines(
    text: str,
    labels: set[str],
    placeholder: str,
) -> list[str]:
    """Return labeled lines whose value is not the expected placeholder."""

    outputs: list[str] = []
    for line in text.splitlines():
        label, separator, value = line.partition(":")
        if not separator:
            continue
        if label.strip().lower() not in labels:
            continue
        if value.strip() != placeholder:
            outputs.append(line)
    return outputs
