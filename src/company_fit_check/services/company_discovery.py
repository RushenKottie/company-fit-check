"""Company discovery from interpreted search criteria."""

from langchain_core.messages import HumanMessage, SystemMessage
from mlflow.entities.span import SpanType
from pydantic import BaseModel, Field

from company_fit_check.llm.client import create_azure_chat_model
from company_fit_check.logging_utils import get_logger
from company_fit_check.models.state import CompanyCandidate, CompanySearchCriteria
from company_fit_check.services.mlflow_tracking import log_llm_prompt_artifact, traced_operation

logger = get_logger(__name__)


class CompanyDiscoveryResult(BaseModel):
    """Structured list of discovered companies."""

    companies: list[CompanyCandidate] = Field(default_factory=list)


class ZeroCompanyClarification(BaseModel):
    """User-facing explanation and question when discovery returns no companies."""

    user_message: str


def discover_companies(
    company_search_criteria: CompanySearchCriteria,
    limit: int = 1000,
) -> list[CompanyCandidate]:
    """Discover companies that plausibly match the interpreted criteria."""

    llm = create_azure_chat_model()
    if llm is None:
        raise RuntimeError("Azure OpenAI is not configured for company discovery.")

    structured_llm = llm.with_structured_output(CompanyDiscoveryResult)
    criteria_text = _format_criteria(company_search_criteria)

    logger.info(
        "LLM call start: discover_companies limit=%s criteria_fields=%s",
        limit,
        sum(1 for value in company_search_criteria.model_dump().values() if value),
    )
    with traced_operation(
        "llm.discover_companies",
        span_type=SpanType.RETRIEVER,
        inputs={
            "limit": limit,
            "criteria_fields": sum(1 for value in company_search_criteria.model_dump().values() if value),
        },
    ) as span:
        messages = [
            SystemMessage(
                content=(
                    "All inputs in this conversation are related to professional "
                    "employment and work preferences. Interpret them only in that "
                    "context. "
                    "You discover companies that plausibly match a structured job-search "
                    "request. Return up to the requested number of companies. For each "
                    "company include: name, website or LinkedIn URL, industry, company "
                    "size, a short discovery reason, and a confidence between 0 and 1. "
                    "Use only the company search criteria provided. These criteria are "
                    "absolute or near-absolute company-side search filters such as "
                    "location, size, domain, stage, role family, or other explicit "
                    "company attributes. Use them only to narrow and identify the "
                    "company set. Do not use matching axes, user-fit dimensions, or "
                    "subjective interpretation in discovery."
                )
            ),
            HumanMessage(
                content=(
                    f"Find up to {limit} companies.\n\n"
                    f"Company search criteria:\n{criteria_text}"
                )
            ),
        ]
        log_llm_prompt_artifact("llm-prompt-discover-companies", messages)
        result = structured_llm.invoke(messages)
        if span is not None:
            span.set_outputs(
                {
                    "company_count": len(result.companies),
                    "companies": [company.model_dump() for company in result.companies],
                }
            )
    logger.info("LLM call end: discover_companies returned=%s", len(result.companies))
    return result.companies


def build_zero_company_clarification_message(
    company_search_criteria: CompanySearchCriteria,
) -> str:
    """Explain why discovery may have failed and ask the user for clarification."""

    llm = create_azure_chat_model()
    if llm is None:
        return (
            "I could not find any companies for the current search criteria. "
            "Please clarify which criteria can be relaxed, broadened, or corrected."
        )

    structured_llm = llm.with_structured_output(ZeroCompanyClarification)
    criteria_text = _format_criteria(company_search_criteria)

    logger.info("LLM call start: build_zero_company_clarification_message")
    with traced_operation(
        "llm.build_zero_company_clarification",
        span_type=SpanType.LLM,
        inputs={"criteria_fields": sum(1 for value in company_search_criteria.model_dump().values() if value)},
    ) as span:
        messages = [
            SystemMessage(
                content=(
                    "All inputs in this conversation are related to professional "
                    "employment and work preferences. Interpret them only in that "
                    "context. "
                    "You explain why a company search may have returned zero results. "
                    "Write a concise user-facing message that identifies the likely "
                    "problem in the search criteria and asks focused follow-up "
                    "questions so the search can be retried."
                )
            ),
            HumanMessage(content=f"Company search criteria:\n{criteria_text}"),
        ]
        log_llm_prompt_artifact("llm-prompt-zero-company-clarification", messages)
        result = structured_llm.invoke(messages)
        if span is not None:
            span.set_outputs({"user_message": result.user_message})
    logger.info("LLM call end: build_zero_company_clarification_message")
    return result.user_message


def refine_company_search_criteria(
    company_search_criteria: CompanySearchCriteria,
    user_clarification: str,
) -> CompanySearchCriteria:
    """Update company search criteria based on the user's follow-up clarification."""

    llm = create_azure_chat_model()
    if llm is None:
        return company_search_criteria

    structured_llm = llm.with_structured_output(CompanySearchCriteria)
    criteria_text = _format_criteria(company_search_criteria)

    logger.info(
        "LLM call start: refine_company_search_criteria current_fields=%s clarification_chars=%s",
        sum(1 for value in company_search_criteria.model_dump().values() if value),
        len(user_clarification.strip()),
    )
    with traced_operation(
        "llm.refine_company_search",
        span_type=SpanType.LLM,
        inputs={
            "criteria_fields": sum(1 for value in company_search_criteria.model_dump().values() if value),
            "clarification_chars": len(user_clarification.strip()),
        },
    ) as span:
        messages = [
            SystemMessage(
                content=(
                    "All inputs in this conversation are related to professional "
                    "employment and work preferences. Interpret them only in that "
                    "context. "
                    "You revise structured company search criteria using the user's "
                    "clarification. Keep useful existing criteria unless the user "
                    "explicitly broadens, narrows, or corrects them."
                )
            ),
            HumanMessage(
                content=(
                    f"Current company search criteria:\n{criteria_text}\n\n"
                    f"User clarification:\n{user_clarification.strip()}"
                )
            ),
        ]
        log_llm_prompt_artifact("llm-prompt-refine-company-search", messages)
        result = structured_llm.invoke(messages)
        if span is not None:
            span.set_outputs(result.model_dump())
    logger.info(
        "LLM call end: refine_company_search_criteria updated_fields=%s",
        sum(1 for value in result.model_dump().values() if value),
    )
    return result


def _format_criteria(company_search_criteria: CompanySearchCriteria) -> str:
    """Format the interpreted company-search criteria for prompting."""

    lines: list[str] = []
    for key, value in company_search_criteria.model_dump().items():
        if value:
            lines.append(f"- {key}: {', '.join(value)}")
    return "\n".join(lines) or "- no explicit criteria provided"
