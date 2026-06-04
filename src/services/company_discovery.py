"""Company discovery from interpreted search criteria."""

import re

from langchain_core.messages import HumanMessage, SystemMessage
from mlflow.entities.span import SpanType
from pydantic import BaseModel, Field

from llm.client import create_azure_chat_model
from logging_utils import get_logger
from models.state import CompanyCandidate, CompanySearchCriteria, HardFilter
from services.mlflow_tracking import log_llm_prompt_artifact, traced_operation

logger = get_logger(__name__)


class CompanyDiscoveryResult(BaseModel):
    """Structured list of discovered companies."""

    companies: list[CompanyCandidate] = Field(default_factory=list)


class ZeroCompanyClarification(BaseModel):
    """User-facing explanation and question when discovery returns no companies."""

    user_message: str


def discover_companies(
    company_search_criteria: CompanySearchCriteria,
    limit: int = 50,
) -> list[CompanyCandidate]:
    """Discover companies that plausibly match the interpreted criteria."""

    llm = create_azure_chat_model()
    if llm is None:
        raise RuntimeError("Azure OpenAI is not configured for company discovery.")

    structured_llm = llm.with_structured_output(CompanyDiscoveryResult)
    criteria_text = _format_criteria(company_search_criteria)
    resolved_limit = _resolve_discovery_limit(company_search_criteria, limit)
    result_count_instruction = _build_result_count_instruction(
        company_search_criteria,
        resolved_limit,
    )

    logger.info(
        "LLM call start: discover_companies limit=%s criteria_fields=%s",
        resolved_limit,
        sum(1 for value in company_search_criteria.model_dump().values() if value),
    )
    with traced_operation(
        "llm.discover_companies",
        span_type=SpanType.RETRIEVER,
        inputs={
            "limit": resolved_limit,
            "criteria_fields": sum(1 for value in company_search_criteria.model_dump().values() if value),
            "hard_filters": [
                filter_.model_dump()
                for filter_ in company_search_criteria.hard_filters
            ],
        },
    ) as span:
        messages = [
            SystemMessage(
                content=(
                    "All inputs in this conversation are related to professional "
                    "employment and work preferences. Interpret them only in that "
                    "context. "
                    "You discover companies that plausibly match a structured job-search "
                    "request. Return the requested number of companies. For each "
                    "company include: name, website or LinkedIn URL, location, industry, "
                    "company size, a short discovery reason, and a confidence between 0 "
                    "and 1. Location must be the company headquarters or primary "
                    "location formatted as City, Country. "
                    "Use only the company search criteria provided. These criteria are "
                    "absolute or near-absolute company-side search filters such as "
                    "location, size, domain, stage, role family, or other explicit "
                    "company attributes. Use them only to narrow and identify the "
                    "company set. Do not use matching axes, user-fit dimensions, or "
                    "subjective interpretation in discovery. "
                    "Hard filters inside company_search_criteria.hard_filters are "
                    "mandatory constraints, not preferences. Preserve and obey their "
                    "literal values. Result-set hard filters control the number of "
                    "companies to return. Company-candidate hard filters such as "
                    "founded date, employee count, valuation, revenue, net worth, or "
                    "funding amount must be used as factual filters for every returned "
                    "company. If the evidence is uncertain, prefer companies that "
                    "publicly and plausibly satisfy the hard filter, and mention the "
                    "hard-filter match in the discovery reason. "
                    "Deduplicate aggressively. Return one row per real employer only. "
                    "Do not include duplicate entries for the same company under "
                    "alternate punctuation, spacing, abbreviations, product brands, "
                    "legacy brands, subsidiaries, or legal-entity suffixes such as "
                    "Ltd, LLC, Inc, GmbH, BV, Plc, Corp, or similar. If multiple "
                    "labels refer to the same employer, choose the clearest canonical "
                    "employer name and output only that one."
                )
            ),
            HumanMessage(
                content=(
                    f"{result_count_instruction}\n\n"
                    f"Company search criteria:\n{criteria_text}"
                )
            ),
        ]
        log_llm_prompt_artifact("llm-prompt-discover-companies", messages)
        result = structured_llm.invoke(messages)
        deduped_companies = _apply_result_count_cap(
            _dedupe_companies_by_website(result.companies),
            company_search_criteria,
        )
        if span is not None:
            span.set_outputs(
                {
                    "raw_company_count": len(result.companies),
                    "company_count": len(deduped_companies),
                    "companies": [company.model_dump() for company in deduped_companies],
                }
            )
    logger.info(
        "LLM call end: discover_companies returned=%s deduped=%s",
        len(result.companies),
        len(deduped_companies),
    )
    return deduped_companies


def get_result_count_hard_filter_violation(
    company_search_criteria: CompanySearchCriteria,
    company_count: int,
) -> HardFilter | None:
    """Return a result-count filter that the current candidate list violates."""

    for hard_filter in _result_count_hard_filters(company_search_criteria):
        count_range = _parse_result_count_range(hard_filter)
        if count_range is None:
            continue
        minimum, maximum = count_range
        if minimum is not None and company_count < minimum:
            return hard_filter
        if maximum is not None and company_count > maximum:
            return hard_filter
    return None


def build_hard_filter_clarification_message(
    hard_filter: HardFilter,
    company_count: int,
) -> str:
    """Build a focused clarification when a mandatory hard filter cannot be met."""

    return (
        f"I found {company_count} companies, but the hard filter "
        f"'{hard_filter.text}' could not be satisfied. Please clarify whether this "
        "filter should be relaxed, changed, or kept exactly as written."
    )


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
                    "explicitly broadens, narrows, or corrects them. Preserve "
                    "company_search_criteria.hard_filters unless the clarification "
                    "explicitly relaxes, changes, or removes one. If the clarification "
                    "adds a concrete numerical, date, money, count, threshold, or range "
                    "requirement, put it in hard_filters as a mandatory filter."
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
        if not _clarification_changes_hard_filters(user_clarification):
            result.hard_filters = _merge_hard_filters(
                company_search_criteria.hard_filters,
                result.hard_filters,
            )
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
        if key == "hard_filters" and company_search_criteria.hard_filters:
            filter_lines = [
                (
                    f"  - text={filter_.text}; scope={filter_.scope}; "
                    f"operator={filter_.operator}; value={filter_.value}"
                )
                for filter_ in company_search_criteria.hard_filters
            ]
            lines.append("- hard_filters:\n" + "\n".join(filter_lines))
        elif value:
            lines.append(f"- {key}: {', '.join(value)}")
    return "\n".join(lines) or "- no explicit criteria provided"


def _resolve_discovery_limit(
    company_search_criteria: CompanySearchCriteria,
    default_limit: int,
) -> int:
    """Use result-set hard filters to choose a discovery limit."""

    resolved_limit = default_limit
    for hard_filter in _result_count_hard_filters(company_search_criteria):
        count_range = _parse_result_count_range(hard_filter)
        if count_range is None:
            continue
        minimum, maximum = count_range
        if maximum is not None:
            resolved_limit = min(resolved_limit, maximum)
        elif minimum is not None:
            resolved_limit = max(resolved_limit, minimum)
    return max(1, resolved_limit)


def _build_result_count_instruction(
    company_search_criteria: CompanySearchCriteria,
    limit: int,
) -> str:
    hard_filter = _first_result_count_hard_filter(company_search_criteria)
    if hard_filter is None:
        return f"Find up to {limit} companies."
    count_range = _parse_result_count_range(hard_filter)
    minimum, maximum = count_range or (None, None)
    if hard_filter.operator == "exactly":
        return f"Find exactly {minimum or limit} companies."
    if hard_filter.operator == "at_most":
        return f"Find at most {maximum or limit} companies."
    if hard_filter.operator == "at_least":
        return f"Find at least {minimum or limit} companies if possible."
    if hard_filter.operator == "between":
        if minimum is not None and maximum is not None:
            return f"Find between {minimum} and {maximum} companies."
        return f"Find between {hard_filter.value} companies."
    return f"Find up to {limit} companies while satisfying all hard filters."


def _apply_result_count_cap(
    companies: list[CompanyCandidate],
    company_search_criteria: CompanySearchCriteria,
) -> list[CompanyCandidate]:
    """Trim candidates when a result-set hard filter defines a maximum."""

    hard_filter = _first_result_count_hard_filter(company_search_criteria)
    if hard_filter is None:
        return companies

    count_range = _parse_result_count_range(hard_filter)
    if count_range is None:
        return companies
    _, maximum = count_range
    if maximum is None:
        return companies
    return companies[:maximum]


def _result_count_hard_filters(
    company_search_criteria: CompanySearchCriteria,
) -> list[HardFilter]:
    return [
        hard_filter
        for hard_filter in company_search_criteria.hard_filters
        if hard_filter.scope == "result_set"
    ]


def _first_result_count_hard_filter(
    company_search_criteria: CompanySearchCriteria,
) -> HardFilter | None:
    filters = _result_count_hard_filters(company_search_criteria)
    return filters[0] if filters else None


def _parse_result_count_range(
    hard_filter: HardFilter,
) -> tuple[int | None, int | None] | None:
    if hard_filter.operator == "between":
        match = re.search(r"(?P<lower>\d+)\D+(?P<upper>\d+)", hard_filter.value)
        if not match:
            return None
        lower = int(match.group("lower"))
        upper = int(match.group("upper"))
        return min(lower, upper), max(lower, upper)

    value = _parse_positive_int(hard_filter.value)
    if value is None:
        return None
    if hard_filter.operator == "exactly":
        return value, value
    if hard_filter.operator == "at_least":
        return value, None
    if hard_filter.operator == "at_most":
        return None, value
    return None


def _parse_positive_int(value: str) -> int | None:
    match = re.search(r"\d+", value)
    if not match:
        return None
    parsed = int(match.group(0))
    return parsed if parsed > 0 else None


def _clarification_changes_hard_filters(user_clarification: str) -> bool:
    normalized = user_clarification.lower()
    return bool(
        re.search(
            r"\b(relax|remove|drop|ignore|change|replace|instead|any\s+number|"
            r"doesn'?t\s+matter|no\s+limit)\b",
            normalized,
        )
    )


def _merge_hard_filters(
    existing: list[HardFilter],
    additions: list[HardFilter],
) -> list[HardFilter]:
    merged: list[HardFilter] = []
    seen: set[tuple[str, str, str, str]] = set()
    for filter_ in [*existing, *additions]:
        key = (
            filter_.text.strip().lower(),
            filter_.scope,
            filter_.operator,
            filter_.value.strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(filter_)
    return merged


def _dedupe_companies_by_website(companies: list[CompanyCandidate]) -> list[CompanyCandidate]:
    """Drop later company rows that share the exact same normalized website/link."""

    deduped_companies: list[CompanyCandidate] = []
    seen_websites: set[str] = set()

    for company in companies:
        website = _normalize_company_locator(company.website_or_linkedin)
        if not website:
            deduped_companies.append(company)
            continue
        if website in seen_websites:
            continue
        seen_websites.add(website)
        deduped_companies.append(company)
    return deduped_companies


def _normalize_company_locator(value: str) -> str:
    """Normalize one website or LinkedIn locator for exact duplicate comparison."""

    return value.strip().rstrip("/").lower()
