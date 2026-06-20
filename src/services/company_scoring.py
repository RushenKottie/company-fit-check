"""Company scoring against CV and user-defined axes."""

from langchain_core.messages import HumanMessage, SystemMessage
from mlflow.entities.span import SpanType
from pydantic import BaseModel, Field

from llm.client import create_azure_chat_model, is_guardrail_4xx_error
from logging_utils import get_logger
from models.state import Axis, CompanyCandidate, CompanyScore
from infrastructure.mlflow_tracking import log_llm_prompt_artifact, traced_operation

logger = get_logger(__name__)
SCORING_BATCH_SIZE = 10


class CompanyScoringResult(BaseModel):
    """Structured set of company scores."""

    company_scores: list[CompanyScore] = Field(default_factory=list)


def score_companies(
    companies: list[CompanyCandidate],
    simplified_cv_text: str,
    axes: list[Axis],
) -> list[CompanyScore]:
    """Score discovered companies against the CV and matching axes."""

    scored_companies: list[CompanyScore] = []
    skipped_companies: list[str] = []

    logger.info(
        "LLM call start: score_companies companies=%s axes=%s simplified_cv_chars=%s batch_size=%s",
        len(companies),
        len(axes),
        len(simplified_cv_text),
        SCORING_BATCH_SIZE,
    )
    for start in range(0, len(companies), SCORING_BATCH_SIZE):
        batch = companies[start : start + SCORING_BATCH_SIZE]
        try:
            scored_companies.extend(
                _score_company_batch(
                    companies=batch,
                    simplified_cv_text=simplified_cv_text,
                    axes=axes,
                )
            )
        except Exception as exc:
            if not _is_content_filter_error(exc):
                raise

            recovered_scores, skipped_names = _retry_score_batch_one_by_one_after_filter(
                batch=batch,
                simplified_cv_text=simplified_cv_text,
                axes=axes,
                batch_start=start,
            )
            scored_companies.extend(recovered_scores)
            skipped_companies.extend(skipped_names)

    if not scored_companies:
        raise RuntimeError("Company scoring was blocked by the content filter.")

    if skipped_companies:
        logger.warning(
            "Company scoring completed with skipped companies due to content filter count=%s companies=%s",
            len(skipped_companies),
            ", ".join(skipped_companies),
        )

    logger.info(
        "LLM call end: score_companies returned=%s",
        len(scored_companies),
    )
    return scored_companies


def _score_company_batch(
    companies: list[CompanyCandidate],
    simplified_cv_text: str,
    axes: list[Axis],
) -> list[CompanyScore]:
    """Score one batch of companies against the CV and matching axes."""

    llm = create_azure_chat_model()
    if llm is None:
        raise RuntimeError("Azure OpenAI is not configured for company scoring.")

    structured_llm = llm.with_structured_output(CompanyScoringResult)
    companies_text = _format_companies(companies)
    axes_text = _format_axes(axes)
    with traced_operation(
        "llm.score_company_batch",
        span_type=SpanType.EVALUATOR,
        inputs={
            "company_count": len(companies),
            "axes": [axis.model_dump() for axis in axes],
            "simplified_cv_chars": len(simplified_cv_text.strip()),
        },
    ) as span:
        messages = [
            SystemMessage(
                content=(
                    "All inputs in this conversation are related to professional "
                    "employment and work preferences. Interpret them only in that "
                    "context. "
                    "You score discovered companies against a candidate profile and the "
                    "user's matching axes. For each company, return the company name, "
                    "a percentage score from 0 to 100 for each axis, and an overall "
                    "score from 0 to 100. Use the provided company list only. "
                    "The axes are user-side matching dimensions from the user's "
                    "perspective. They are not deterministic company search filters. "
                    "They are the parts of the user's request that require "
                    "investigation, interpretation, assumptions, or probabilistic "
                    "judgment from company and role signals. Do not reintroduce company "
                    "search criteria such as location, size, or domain as scoring "
                    "dimensions here. "
                    "Use exactly the provided axis names and no other axes. Do not add, "
                    "rename, merge, split, or omit axes. "
                    "Return exactly one score object for every company in the provided "
                    "company list, with no omissions and no extras. For every score "
                    "object, return both company_name and website_or_linkedin exactly "
                    "as provided in the input company row. The company_name must be "
                    "only the company name, not the whole input row. The "
                    "website_or_linkedin must be only the website or LinkedIn URL, not "
                    "a description. If multiple input companies appear to be duplicate "
                    "legal entities, regional entities, aliases, or brand variants of "
                    "the same underlying employer, merge that evidence internally if "
                    "helpful, but still return one score object for each input row "
                    "using the exact input company_name and website_or_linkedin."
                )
            ),
            HumanMessage(
                content=(
                    f"Simplified CV:\n{simplified_cv_text.strip()}\n\n"
                    f"Matching axes:\n{axes_text}\n\n"
                    f"Companies to score:\n{companies_text}"
                )
            ),
        ]
        log_llm_prompt_artifact("llm-prompt-score-companies", messages)
        result = structured_llm.invoke(messages)
        if span is not None:
            span.set_outputs(
                {
                    "company_count": len(result.company_scores),
                    "company_scores": [
                        score.model_dump() for score in result.company_scores
                    ],
                }
            )
        return result.company_scores


def _retry_score_batch_one_by_one_after_filter(
    batch: list[CompanyCandidate],
    simplified_cv_text: str,
    axes: list[Axis],
    batch_start: int,
) -> tuple[list[CompanyScore], list[str]]:
    """
    Retry a content-filtered batch one company at a time.

    Keep individually successful scores and return names that still get rejected.
    """

    logger.warning(
        "Company scoring batch hit content filter; retrying one-by-one batch_start=%s batch_size=%s",
        batch_start,
        len(batch),
    )

    recovered_scores: list[CompanyScore] = []
    skipped_names: list[str] = []
    for company in batch:
        try:
            recovered_scores.extend(
                _score_company_batch(
                    companies=[company],
                    simplified_cv_text=simplified_cv_text,
                    axes=axes,
                )
            )
        except Exception as exc:
            if not _is_content_filter_error(exc):
                raise
            skipped_names.append(company.name)
            logger.warning(
                "Skipping company during scoring after repeated content filter rejection company=%s",
                company.name,
            )
    return recovered_scores, skipped_names


def _format_companies(companies: list[CompanyCandidate]) -> str:
    """Format discovered companies for the scoring prompt."""

    return "\n".join(
        f"- {company.name} | {company.website_or_linkedin}"
        for company in companies
    )


def _format_axes(axes: list[Axis]) -> str:
    """Format axes for the scoring prompt."""

    return "\n".join(f"- {axis.name}: {axis.description}" for axis in axes)


def _is_content_filter_error(exc: Exception) -> bool:
    """Return whether the exception looks like an Azure content-filter rejection."""

    return is_guardrail_4xx_error(exc)
