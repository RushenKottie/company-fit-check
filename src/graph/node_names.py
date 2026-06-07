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
    REFINE_COMPANY_SEARCH = "refine_company_search"
    SEARCH_COMPANIES = "search_companies"
    SCORE_COMPANIES = "score_companies"

    @property
    def span_name(self) -> str:
        """Return the MLflow span name emitted for this graph node."""

        return f"node.{self.value}"


class WorkflowRouteName(StrEnum):
    """Stable names for non-node graph route outcomes."""

    STOP = "stop"
