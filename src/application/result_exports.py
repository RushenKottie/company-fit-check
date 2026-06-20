"""Export completed workflow results into CSV files."""

import csv
import io
import re

from logging_utils import get_logger
from models.artifacts import GeneratedCsvArtifact
from models.state import (
    SESSION_STATUS_COMPLETED,
    CompanyFitState,
    FinalCompanyResult,
)

logger = get_logger(__name__)

RESULT_CSV_BASE_COLUMNS = tuple(
    field_name
    for field_name in FinalCompanyResult.model_fields
    if field_name != "axis_scores"
)


def build_results_csv(state: CompanyFitState) -> GeneratedCsvArtifact:
    """Build a CSV artifact from a completed workflow state."""

    if state.get("session_status") != SESSION_STATUS_COMPLETED:
        raise ValueError("CSV export is only available for completed workflows.")

    final_results = [
        FinalCompanyResult.model_validate(result)
        for result in state.get("final_results", [])
    ]
    fieldnames = build_results_csv_fieldnames(final_results)

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for result in final_results:
        writer.writerow(_build_csv_row(result, fieldnames))

    logger.info(
        "Built CSV artifact rows=%s columns=%s",
        len(final_results),
        len(fieldnames),
    )
    return GeneratedCsvArtifact(
        filename="company-fit-results.csv",
        content_type="text/csv",
        content_bytes=buffer.getvalue().encode("utf-8"),
    )


def build_results_csv_fieldnames(
    final_results: list[FinalCompanyResult],
) -> list[str]:
    """Return the base CSV fields plus one score column per axis."""

    axis_scores = final_results[0].axis_scores if final_results else []
    return [
        *RESULT_CSV_BASE_COLUMNS,
        *[
            _build_axis_column_name(index=index, axis_name=axis.axis_name)
            for index, axis in enumerate(axis_scores, start=1)
        ],
    ]


def _build_csv_row(
    result: FinalCompanyResult,
    fieldnames: list[str],
) -> dict[str, object]:
    """Return one flattened CSV row with dynamic axis score columns."""

    row = result.model_dump(exclude={"axis_scores"})
    axis_columns = fieldnames[len(RESULT_CSV_BASE_COLUMNS) :]
    for axis_score, column_name in zip(result.axis_scores, axis_columns, strict=False):
        row[column_name] = axis_score.score
    return row


def _build_axis_column_name(index: int, axis_name: str) -> str:
    """Generate a stable CSV column name for one axis score."""

    slug = re.sub(r"[^a-z0-9]+", "_", axis_name.lower()).strip("_")
    if not slug:
        slug = f"axis_{index}"
    return f"axis_{index}_{slug}_score"
