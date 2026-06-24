"""Shared LangGraph node and route names."""

from enum import StrEnum


class WorkflowNodeName(StrEnum):
    """Stable names for workflow graph nodes."""

    ENTRY = "entry"
    EXTRACT_AND_MASK_CV = "extract_and_mask_cv"
    VALIDATE_PII_MASKING = "validate_pii_masking"
    SIMPLIFY_CV = "simplify_cv"
    INTERPRET_USER_INPUT = "interpret_user_input"
    VALIDATE_USER_INPUT_INTERPRETATION = "validate_user_input_interpretation"
    SEARCH_COMPANIES = "search_companies"
    SCORE_COMPANIES = "score_companies"
    PREPARE_FINAL_RESULTS = "prepare_final_results"


STOP_ROUTE = "stop"
