"""MLflow scorers for deterministic evaluation cases."""

from __future__ import annotations

from mlflow.genai.scorers import scorer

from evals.checks import CHECK_FUNCTIONS
from evals.models import CaseExecutionResult


def _score_from_check(check_name: str, outputs, inputs) -> bool:
    result = CaseExecutionResult.model_validate(outputs)
    return CHECK_FUNCTIONS[check_name](result, inputs or {}).passed


@scorer(name="llm_uses_masked_cv_not_raw_cv")
def score_llm_uses_masked_cv_not_raw_cv(inputs, outputs) -> bool:
    return _score_from_check("llm_uses_masked_cv_not_raw_cv", outputs, inputs)


@scorer(name="masked_cv_has_no_contact_details")
def score_masked_cv_has_no_contact_details(inputs, outputs) -> bool:
    return _score_from_check("masked_cv_has_no_contact_details", outputs, inputs)


@scorer(name="workflow_stops_on_masking_failure")
def score_workflow_stops_on_masking_failure(inputs, outputs) -> bool:
    return _score_from_check("workflow_stops_on_masking_failure", outputs, inputs)


@scorer(name="company_search_blocked_until_prereqs_valid")
def score_company_search_blocked_until_prereqs_valid(inputs, outputs) -> bool:
    return _score_from_check("company_search_blocked_until_prereqs_valid", outputs, inputs)


@scorer(name="clarification_loop_is_bounded")
def score_clarification_loop_is_bounded(inputs, outputs) -> bool:
    return _score_from_check("clarification_loop_is_bounded", outputs, inputs)


@scorer(name="guardrail_4xx_requests_rephrase")
def score_guardrail_4xx_requests_rephrase(inputs, outputs) -> bool:
    return _score_from_check("guardrail_4xx_requests_rephrase", outputs, inputs)


@scorer(name="guardrail_4xx_rephrase_stops_workflow")
def score_guardrail_4xx_rephrase_stops_workflow(inputs, outputs) -> bool:
    return _score_from_check("guardrail_4xx_rephrase_stops_workflow", outputs, inputs)


@scorer(name="prompt_rephrase_replaces_blocked_prompt")
def score_prompt_rephrase_replaces_blocked_prompt(inputs, outputs) -> bool:
    return _score_from_check("prompt_rephrase_replaces_blocked_prompt", outputs, inputs)


@scorer(name="guardrail_4xx_fails_with_restart")
def score_guardrail_4xx_fails_with_restart(inputs, outputs) -> bool:
    return _score_from_check("guardrail_4xx_fails_with_restart", outputs, inputs)


@scorer(name="guardrail_4xx_fails_with_cv_cleanup")
def score_guardrail_4xx_fails_with_cv_cleanup(inputs, outputs) -> bool:
    return _score_from_check("guardrail_4xx_fails_with_cv_cleanup", outputs, inputs)


@scorer(name="generic_llm_failure_is_sanitized")
def score_generic_llm_failure_is_sanitized(inputs, outputs) -> bool:
    return _score_from_check("generic_llm_failure_is_sanitized", outputs, inputs)


@scorer(name="score_shape_is_valid_for_all_companies")
def score_score_shape_is_valid_for_all_companies(inputs, outputs) -> bool:
    return _score_from_check("score_shape_is_valid_for_all_companies", outputs, inputs)


@scorer(name="csv_schema_is_valid")
def score_csv_schema_is_valid(inputs, outputs) -> bool:
    return _score_from_check("csv_schema_is_valid", outputs, inputs)


ALL_SCORERS = [
    score_llm_uses_masked_cv_not_raw_cv,
    score_masked_cv_has_no_contact_details,
    score_workflow_stops_on_masking_failure,
    score_company_search_blocked_until_prereqs_valid,
    score_clarification_loop_is_bounded,
    score_guardrail_4xx_requests_rephrase,
    score_guardrail_4xx_rephrase_stops_workflow,
    score_prompt_rephrase_replaces_blocked_prompt,
    score_guardrail_4xx_fails_with_restart,
    score_guardrail_4xx_fails_with_cv_cleanup,
    score_generic_llm_failure_is_sanitized,
    score_score_shape_is_valid_for_all_companies,
    score_csv_schema_is_valid,
]
