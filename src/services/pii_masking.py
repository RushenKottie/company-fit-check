"""Local PII masking before any external LLM call."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re

os.environ["TLDEXTRACT_CACHE"] = str(
    Path(__file__).resolve().parents[2] / ".tmp" / "tldextract-cache"
)

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from logging_utils import get_logger

logger = get_logger(__name__)
CONTACT_LINK_PATTERNS = [
    re.compile(r"https?://(?:www\.)?linkedin\.com/in/[^\s)]+", flags=re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?github\.com/[^\s/]+/?", flags=re.IGNORECASE),
    re.compile(r"mailto:[^\s]+", flags=re.IGNORECASE),
]
ADDRESS_LINE_PATTERNS = [
    re.compile(r"(?im)^(address|home address|residential address)\s*:\s*.+$"),
]
BIRTH_DATE_LINE_PATTERNS = [
    re.compile(r"(?im)^(date of birth|birth date|dob)\s*:\s*.+$"),
]


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
    sanitized_text = anonymized.text
    for pattern in CONTACT_LINK_PATTERNS:
        sanitized_text = pattern.sub("<CONTACT_LINK>", sanitized_text)
    for pattern in ADDRESS_LINE_PATTERNS:
        sanitized_text = pattern.sub("Address: <ADDRESS>", sanitized_text)
    for pattern in BIRTH_DATE_LINE_PATTERNS:
        sanitized_text = pattern.sub("Date of birth: <DATE_OF_BIRTH>", sanitized_text)

    logger.info("Completed local PII masking output_chars=%s", len(sanitized_text))
    return PiiMaskingResult(text=sanitized_text)
