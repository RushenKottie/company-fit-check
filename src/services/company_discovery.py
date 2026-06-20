"""Company discovery from interpreted search criteria."""

from langchain_core.messages import HumanMessage, SystemMessage
from mlflow.entities.span import SpanType
from pydantic import BaseModel, Field

from llm.client import create_azure_chat_model
from logging_utils import get_logger
from models.state import CompanyCandidate, CompanySearchCriteria, HardFilter
from infrastructure.mlflow_tracking import log_llm_prompt_artifact, traced_operation

logger = get_logger(__name__)


class CompanyDiscoveryResult(BaseModel):
    """Structured list of discovered companies."""

    companies: list[CompanyCandidate] = Field(default_factory=list)


def discover_companies(
    company_search_criteria: CompanySearchCriteria,
    limit: int = 50,
) -> list[CompanyCandidate]:
    """Discover companies that likely match the interpreted criteria."""

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
                    "You discover companies that likely match a structured job-search "
                    "request. Return candidate companies according to the result-count "
                    "instruction. For each "
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
                    "literal values. Result-set hard filters control final result "
                    "counts after scoring. Company-candidate hard filters such as "
                    "founded date, employee count, valuation, revenue, net worth, or "
                    "funding amount must be used as factual filters for every returned "
                    "company. If the evidence is uncertain, prefer companies that "
                    "publicly seem to satisfy the hard filter, and mention the "
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
        deduped_companies = _dedupe_companies_by_website(result.companies)
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


def get_result_count_maximum(
    company_search_criteria: CompanySearchCriteria,
) -> int | None:
    """Return the max number of final companies the user requested, if any."""

    maximums: list[int] = []
    for hard_filter in _result_count_hard_filters(company_search_criteria):
        if hard_filter.maximum is not None:
            maximums.append(hard_filter.maximum)
    return min(maximums) if maximums else None


def _format_criteria(company_search_criteria: CompanySearchCriteria) -> str:
    """Format the interpreted company-search criteria for prompting."""

    lines: list[str] = []
    for key, value in company_search_criteria.model_dump().items():
        if key == "hard_filters" and company_search_criteria.hard_filters:
            filter_lines = [
                (
                    f"  - text={filter_.text}; scope={filter_.scope}; "
                    f"operator={filter_.operator}; value={filter_.value}; "
                    f"minimum={filter_.minimum}; maximum={filter_.maximum}"
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
        if hard_filter.minimum is not None:
            resolved_limit = max(resolved_limit, hard_filter.minimum)
    return max(1, resolved_limit)


def _build_result_count_instruction(
    company_search_criteria: CompanySearchCriteria,
    limit: int,
) -> str:
    """Build the company-count instruction for the discovery prompt."""

    hard_filter = next(iter(_result_count_hard_filters(company_search_criteria)), None)
    if hard_filter is None:
        return f"Find up to {limit} companies."
    operator = hard_filter.operator.replace("_", " ")
    if (
        hard_filter.operator == "between"
        and hard_filter.minimum is not None
        and hard_filter.maximum is not None
    ):
        value = f"{hard_filter.minimum} and {hard_filter.maximum}"
    else:
        value = str(hard_filter.maximum or hard_filter.minimum or limit)
    return f"Find {operator} {value} companies."


def _result_count_hard_filters(
    company_search_criteria: CompanySearchCriteria,
) -> list[HardFilter]:
    """Return hard filters that apply to the requested result count."""

    return [
        hard_filter
        for hard_filter in company_search_criteria.hard_filters
        if hard_filter.scope == "result_set"
    ]


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
