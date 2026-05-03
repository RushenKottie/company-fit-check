"""Local PII masking before any external LLM call."""

from __future__ import annotations

from dataclasses import dataclass

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from company_fit_check.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PiiMaskingResult:
    """Result of local PII masking."""

    text: str


def mask_pii_locally(text: str) -> PiiMaskingResult:
    """Mask PII at the place of execution using Presidio."""

    if not text:
        return PiiMaskingResult(text="")

    logger.info("Starting local PII masking input_chars=%s", len(text))
    analyzer = AnalyzerEngine()
    analyzer_results = analyzer.analyze(text=text, language="en")
    logger.info("PII analyzer detected entities=%s", len(analyzer_results))

    anonymizer = AnonymizerEngine()
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators={
            "DEFAULT": OperatorConfig("replace"),
        },
    )
    logger.info("Completed local PII masking output_chars=%s", len(anonymized.text))
    return PiiMaskingResult(text=anonymized.text)
