"""Export completed workflow results into reusable file artifacts."""

import csv
import io
import re

from logging_utils import get_logger
from models.artifacts import GeneratedArtifact
from models.state import CompanyFitState

logger = get_logger(__name__)


def build_results_csv(state: CompanyFitState) -> GeneratedArtifact:
    """Build a CSV artifact from a completed workflow state."""

    if state.get("session_status") != "completed":
        raise ValueError("CSV export is only available for completed workflows.")

    axes = state.get("axes", [])
    companies = state.get("companies", [])
    company_scores = state.get("company_scores", [])
    companies_by_name = {company.name: company for company in companies}
    axis_columns = [
        _build_axis_column_name(index=index, axis_name=axis.name)
        for index, axis in enumerate(axes, start=1)
    ]

    fieldnames = [
        "company_name",
        "website_or_linkedin",
        "industry",
        "company_size",
        "discovery_reason",
        "overall_score",
        *axis_columns,
    ]

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()

    for company_score in company_scores:
        company = companies_by_name.get(company_score.company_name)
        axis_scores_by_name = {
            axis_score.axis: axis_score.percentage
            for axis_score in company_score.axis_scores
        }

        row = {
            "company_name": company_score.company_name,
            "website_or_linkedin": (
                company.website_or_linkedin if company is not None else ""
            ),
            "industry": company.industry if company is not None else "",
            "company_size": company.company_size if company is not None else "",
            "discovery_reason": (
                company.discovery_reason if company is not None else ""
            ),
            "overall_score": company_score.overall_score,
        }

        for axis, column_name in zip(axes, axis_columns, strict=False):
            row[column_name] = axis_scores_by_name.get(axis.name, "")

        writer.writerow(row)

    logger.info(
        "Built CSV artifact rows=%s axis_columns=%s",
        len(company_scores),
        len(axis_columns),
    )
    return GeneratedArtifact(
        filename="company-fit-results.csv",
        content_type="text/csv",
        content_bytes=buffer.getvalue().encode("utf-8"),
    )


def _build_axis_column_name(index: int, axis_name: str) -> str:
    """Generate a stable CSV column name for one axis score."""

    slug = re.sub(r"[^a-z0-9]+", "_", axis_name.lower()).strip("_")
    if not slug:
        slug = f"axis_{index}"
    return f"axis_{index}_{slug}_score"
