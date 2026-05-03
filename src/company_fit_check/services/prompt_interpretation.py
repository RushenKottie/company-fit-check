"""Prompt interpretation into search criteria and axes."""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage
from mlflow.entities.span import SpanType

from company_fit_check.llm.client import create_azure_chat_model
from company_fit_check.logging_utils import get_logger
from company_fit_check.models.state import Axis, PromptInterpretation
from company_fit_check.services.mlflow_tracking import log_llm_prompt_artifact, traced_operation

logger = get_logger(__name__)


def interpret_prompt(
    prompt: str,
    simplified_cv_text: str,
    clarification: str | None = None,
    previous_axes: list[Axis] | None = None,
) -> PromptInterpretation:
    """Interpret the user prompt and simplified CV into structured context."""

    if not prompt.strip():
        raise ValueError("User prompt is empty.")
    if not simplified_cv_text.strip():
        raise ValueError("Simplified CV text is empty.")

    llm = create_azure_chat_model()
    if llm is None:
        raise RuntimeError("Azure OpenAI is not configured for prompt interpretation.")

    clarification_block = (
        f"\nUser clarification / additional context:\n{clarification.strip()}\n"
        if clarification
        else ""
    )
    explicit_axes = _extract_explicit_axes(clarification) or _extract_explicit_axes(prompt)
    explicit_axes_block = ""
    if explicit_axes:
        explicit_axes_block = (
            "\nExplicit user-declared axes that must be preserved as the axes output:\n"
            + "\n".join(f"- {axis}" for axis in explicit_axes)
            + "\n"
        )

    previous_axes_block = ""
    if previous_axes:
        previous_axes_block = (
            "\nExisting axes from the previous turn. Keep their meaning stable unless "
            "the user explicitly changed them:\n"
            + "\n".join(f"- {axis.name}: {axis.description}" for axis in previous_axes)
            + "\n"
        )
    structured_llm = llm.with_structured_output(PromptInterpretation)
    logger.info(
        "LLM call start: interpret_prompt prompt_chars=%s simplified_cv_chars=%s clarification_present=%s explicit_axes=%s previous_axes=%s",
        len(prompt.strip()),
        len(simplified_cv_text.strip()),
        bool(clarification and clarification.strip()),
        len(explicit_axes),
        len(previous_axes or []),
    )
    with traced_operation(
        "llm.interpret_prompt",
        span_type=SpanType.LLM,
        inputs={
            "prompt_chars": len(prompt.strip()),
            "simplified_cv_chars": len(simplified_cv_text.strip()),
            "clarification_present": bool(clarification and clarification.strip()),
            "previous_axes_count": len(previous_axes or []),
            "explicit_axes": explicit_axes,
        },
    ) as span:
        messages = [
            SystemMessage(
                content=(
                    "All inputs in this conversation are related to professional "
                    "employment and work preferences. Interpret them only in that "
                    "context. "
                    "Interpret a company-search prompt into search criteria and axes "
                    "using the requested structured output schema. "
                    "Use this separation rule strictly. "
                    "company_search_criteria must contain only company-side facts that "
                    "have an absolute or near-absolute answer and can be treated like "
                    "deterministic filters for search. These are things such as "
                    "location, company size, industry or domain, company stage, role "
                    "family, work mode, or other administrative, legal, or explicit "
                    "company attributes that can be written down as search filters. "
                    "Axes must contain everything else from the user's prompt that is "
                    "not absolute and needs interpretation, investigation, assumption, "
                    "or probabilistic judgment. If something cannot be answered with "
                    "full confidence from a simple company fact and instead requires "
                    "reasoning about signals, industry practice, likelihood, fit, or "
                    "interpretation, it belongs in axes. "
                    "Axes must come from the user's prompt or clarification, not from "
                    "decomposing the CV into separate dimensions. "
                    "Search criteria are used only for company discovery. They are not "
                    "matching dimensions and should not be restated as axes. "
                    "Examples that usually belong in axes because they require "
                    "investigation or assumption include English-working environment, "
                    "career-switch openness, compensation fit, seniority fit, culture "
                    "fit, transition friendliness, or likely team suitability. "
                    "The CV may be used to understand what an axis means from the "
                    "user's perspective and to help draft an axis description, but it "
                    "must not create new axes that the user did not ask for. "
                    "If the user explicitly names axes, preserve those axes and do not "
                    "invent extra ones. You may moderately refine wording while keeping "
                    "the same meaning. "
                    "If a clarification asks to add an axis, treat that as additive: "
                    "keep the existing axes unless the user explicitly removes or "
                    "replaces them. "
                    "For each axis, capture both the axis name and the intended meaning "
                    "of the axis from the user's perspective. You may use the prompt, "
                    "the clarification, the CV, and normal industry practice to draft "
                    "that meaning. If the meaning is still unclear, leave the "
                    "description empty instead of inventing a confident explanation. "
                    "Do not ask follow-up questions inside the axes output. "
                    "Unless the user explicitly revises the axes, keep prior axes "
                    "semantically stable across turns. "
                    "The company_search_criteria.undefined field is for unusual "
                    "criteria that do not fit the typed fields, such as founder "
                    "traits or other special constraints. "
                    "Axis descriptions should explain the user's perspective when that "
                    "perspective can be inferred."
                )
            ),
            HumanMessage(
                content=(
                    f"User prompt:\n{prompt.strip()}\n\n"
                    f"{previous_axes_block}"
                    f"{explicit_axes_block}"
                    f"Simplified CV:\n{simplified_cv_text.strip()}\n"
                    f"{clarification_block}"
                )
            ),
        ]
        log_llm_prompt_artifact("llm-prompt-interpret-prompt", messages)
        result = structured_llm.invoke(messages)
        if span is not None:
            span.set_outputs(
                {
                    "axes": [axis.model_dump() for axis in result.axes],
                    "company_search_criteria": result.company_search_criteria.model_dump(),
                }
            )
    logger.info(
        "LLM call end: interpret_prompt axes=%s criteria_fields=%s",
        len(result.axes),
        sum(1 for value in result.company_search_criteria.model_dump().values() if value),
    )
    return result


def _extract_explicit_axes(text: str | None) -> list[str]:
    """Extract explicitly declared axes from user text when possible."""

    if not text:
        return []

    stripped = text.strip()
    if not stripped:
        return []

    numbered_axes = [
        match.strip(" -:\t")
        for match in re.findall(
            r"(?:^|\n)\s*\d+[\.\):\-]\s*(.+?)(?=(?:\n\s*\d+[\.\):\-]\s*)|\Z)",
            stripped,
            flags=re.DOTALL,
        )
    ]
    if len(numbered_axes) >= 2:
        return [_normalize_axis_text(axis) for axis in numbered_axes if _normalize_axis_text(axis)]

    conversational_axes = _extract_conversational_axes(stripped)
    if conversational_axes:
        return conversational_axes

    lower = stripped.lower()
    marker = "axes are:"
    if marker in lower:
        remainder = stripped[lower.index(marker) + len(marker):].strip()
        parts = re.split(r",|\band\b", remainder, flags=re.IGNORECASE)
        normalized = [_normalize_axis_text(part) for part in parts]
        return [part for part in normalized if part]

    return []


def _normalize_axis_text(text: str) -> str:
    """Normalize one user-declared axis string."""

    normalized = re.sub(r"\s+", " ", text).strip(" .;,-\n\t")
    return normalized


def _extract_conversational_axes(text: str) -> list[str]:
    """Extract explicit axes phrased conversationally in one sentence."""

    patterns = [
        r"(?:please\s+)?add\s+an?\s+axis\s+for\s+(.+?)(?=$|\n)",
        r"(?:please\s+)?add\s+an?\s+axis\s+capturing\s+(.+?)(?=$|\n)",
        r"(?:please\s+)?include\s+an?\s+axis\s+for\s+(.+?)(?=$|\n)",
        r"(?:please\s+)?include\s+an?\s+axis\s+capturing\s+(.+?)(?=$|\n)",
        r"an?\s+axis\s+for\s+(.+?)(?=$|\n)",
        r"an?\s+axis\s+capturing\s+(.+?)(?=$|\n)",
        r"axis\s+capturing\s+(.+?)(?=$|\n)",
    ]

    extracted: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            normalized = _normalize_axis_text(match)
            if normalized:
                extracted.append(normalized)

    deduped: list[str] = []
    seen: set[str] = set()
    for axis in extracted:
        lowered = axis.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(axis)
    return deduped
