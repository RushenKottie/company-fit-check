"""Company scoring against CV and user-defined axes."""

from statistics import mean

from langchain_core.messages import HumanMessage, SystemMessage
from mlflow.entities.span import SpanType
from pydantic import BaseModel, Field

from company_fit_check.llm.client import create_azure_chat_model
from company_fit_check.logging_utils import get_logger
from company_fit_check.models.state import Axis, CompanyCandidate, CompanyScore
from company_fit_check.services.mlflow_tracking import log_llm_prompt_artifact, traced_operation

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

    normalized_scores: list[CompanyScore] = []
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
            normalized_scores.extend(
                _score_company_batch(
                    companies=batch,
                    simplified_cv_text=simplified_cv_text,
                    axes=axes,
                )
            )
        except Exception as exc:
            if not _is_content_filter_error(exc):
                raise

            logger.warning(
                "Company scoring batch hit content filter; retrying one-by-one batch_start=%s batch_size=%s",
                start,
                len(batch),
            )
            for company in batch:
                try:
                    normalized_scores.extend(
                        _score_company_batch(
                            companies=[company],
                            simplified_cv_text=simplified_cv_text,
                            axes=axes,
                        )
                    )
                except Exception as single_exc:
                    if not _is_content_filter_error(single_exc):
                        raise
                    skipped_companies.append(company.name)
                    logger.warning(
                        "Skipping company during scoring after repeated content filter rejection company=%s",
                        company.name,
                    )

    if not normalized_scores:
        raise RuntimeError("Company scoring was blocked by the content filter.")

    if skipped_companies:
        logger.warning(
            "Company scoring completed with skipped companies due to content filter count=%s companies=%s",
            len(skipped_companies),
            ", ".join(skipped_companies),
        )

    logger.info(
        "LLM call end: score_companies returned=%s normalized=%s",
        len(normalized_scores),
        len(normalized_scores),
    )
    return normalized_scores


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
                    "Axis names in the response must exactly match the provided axes."
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
        normalized = _normalize_company_scores(result.company_scores, axes)
        if span is not None:
            span.set_outputs(
                {
                    "company_count": len(normalized),
                    "company_scores": [score.model_dump() for score in normalized],
                }
            )
        return normalized


def _format_companies(companies: list[CompanyCandidate]) -> str:
    """Format discovered companies for the scoring prompt."""

    return "\n".join(
        (
            f"- {company.name} | {company.website_or_linkedin} | "
            f"{company.industry} | {company.company_size} | "
            f"reason: {company.discovery_reason}"
        )
        for company in companies
    )


def _format_axes(axes: list[Axis]) -> str:
    """Format axes for the scoring prompt."""

    return "\n".join(f"- {axis.name}: {axis.description}" for axis in axes)


def _normalize_company_scores(
    company_scores: list[CompanyScore],
    axes: list[Axis],
) -> list[CompanyScore]:
    """Normalize model-returned scores and compute overall deterministically."""

    axis_names = [axis.name for axis in axes if axis.name.strip()]
    normalized_company_scores: list[CompanyScore] = []

    for company_score in company_scores:
        score_by_axis = {
            axis_score.axis.strip(): _normalize_percentage(axis_score.percentage)
            for axis_score in company_score.axis_scores
            if axis_score.axis and axis_score.axis.strip()
        }

        normalized_axis_scores = []
        for axis_name in axis_names:
            if axis_name in score_by_axis:
                normalized_axis_scores.append(
                    {
                        "axis": axis_name,
                        "percentage": score_by_axis[axis_name],
                    }
                )

        overall_score = mean(
            axis_score["percentage"] for axis_score in normalized_axis_scores
        ) if normalized_axis_scores else 0.0

        normalized_company_scores.append(
            CompanyScore(
                company_name=company_score.company_name,
                axis_scores=normalized_axis_scores,
                overall_score=round(overall_score, 1),
            )
        )

    return normalized_company_scores


def _normalize_percentage(value: float) -> float:
    """Normalize either 0-1 or 0-100 scores to a 0-100 percentage."""

    normalized = value * 100 if 0.0 <= value <= 1.0 else value
    normalized = max(0.0, min(100.0, normalized))
    return round(normalized, 1)


def _is_content_filter_error(exc: Exception) -> bool:
    """Return whether the exception looks like an Azure content-filter rejection."""

    message = str(exc).lower()
    return "content filter" in message or "content_filter" in message
