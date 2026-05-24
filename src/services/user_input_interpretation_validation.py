"""User-input interpretation validation for the current workflow."""

from langchain_core.messages import HumanMessage, SystemMessage
from mlflow.entities.span import SpanType
from pydantic import BaseModel, Field

from llm.client import create_azure_chat_model
from logging_utils import get_logger
from models.state import Axis
from services.mlflow_tracking import log_llm_prompt_artifact, traced_operation

MAX_AXES = 4
AXIS_MERGE_FALLBACK = (
    "I can suggest merging overlapping axes or substituting narrower ones so "
    f"the final list stays within {MAX_AXES}."
)
logger = get_logger(__name__)


class ClarificationAssessment(BaseModel):
    """Validation result for a single clarification domain."""

    needs_clarification: bool = False
    user_message: str = Field(
        description=(
            "A concise message for the user for this domain only."
        )
    )


def validate_user_input_interpretation(
    axes: list[Axis],
    simplified_cv_text: str,
    prompt: str,
    clarification: str | None = None,
) -> tuple[bool, str | None]:
    """Validate the user-input interpretation and ask for clarifications when needed."""

    logger.info(
        "Validating user-input interpretation axes=%s simplified_cv_chars=%s prompt_chars=%s",
        len(axes),
        len(simplified_cv_text),
        len(prompt),
    )
    if not axes:
        return False, "Please provide at least one matching axis."

    axis_issue_message = _build_axis_issue_message(axes)

    axis_assessment = _assess_axis_clarification_needs(
        axes=axes,
        axis_issue_message=axis_issue_message,
        prompt=prompt,
        clarification=clarification,
    )

    cv_assessment = _assess_cv_sufficiency(
        axes=axes,
        simplified_cv_text=simplified_cv_text,
        prompt=prompt,
        clarification=clarification,
    )

    messages = [
        assessment.user_message
        for assessment in (axis_assessment, cv_assessment)
        if assessment is not None and assessment.needs_clarification
    ]
    if messages:
        logger.info(
            "User-input interpretation validation needs clarification message_count=%s",
            len(messages),
        )
        return False, " ".join(messages)

    if axis_issue_message:
        logger.info("User-input interpretation validation failed deterministic axis checks.")
        return False, axis_issue_message

    logger.info("User-input interpretation validation passed without clarification.")
    return True, None


def _build_axis_issue_message(axes: list[Axis]) -> str | None:
    """Collect deterministic axis-only issues before broader clarification checks."""

    axis_messages: list[str] = []
    if len(axes) > MAX_AXES:
        axis_messages.append(_build_axis_limit_message(axes))

    incomplete = [axis.name or "(unnamed axis)" for axis in axes if not axis.description]
    if incomplete:
        axis_messages.append(
            "Each axis needs a short description from your perspective. "
            f"Please clarify: {', '.join(incomplete)}"
        )

    if not axis_messages:
        return None
    return " ".join(axis_messages)


def _build_axis_limit_message(axes: list[Axis]) -> str:
    """Describe the axis-limit issue and ask for a smaller set."""

    names = ", ".join(axis.name for axis in axes if axis.name)
    suggestion = _suggest_axis_merge_or_substitution(axes)
    message = f"You have exceeded the maximum number of axis which is {MAX_AXES}."
    if names:
        message = f"{message} Current axes: {names}."
    if suggestion:
        message = f"{message} {suggestion}"
    return message


def _suggest_axis_merge_or_substitution(axes: list[Axis]) -> str:
    """Ask the LLM for concise merge/substitution suggestions when axes exceed the limit."""

    llm = create_azure_chat_model()
    if llm is None:
        return AXIS_MERGE_FALLBACK

    axis_lines = "\n".join(
        f"- {axis.name or '(unnamed axis)'}: {axis.description or '(no description)'}"
        for axis in axes
    )

    logger.info("LLM call start: suggest_axis_merge_or_substitution axis_count=%s", len(axes))
    with traced_operation(
        "llm.suggest_axis_merge_or_substitution",
        span_type=SpanType.LLM,
        inputs={"axis_count": len(axes)},
    ) as span:
        messages = [
            SystemMessage(
                content=(
                    "All inputs in this conversation are related to professional "
                    "employment and work preferences. Interpret them only in that "
                    "context. "
                    "You help users reduce a list of company-fit matching axes. "
                    "Reply with one concise message that starts exactly with "
                    "'I can suggest...'. Recommend which axes could be merged, "
                    "substituted, or removed to fit the limit. Do not use bullets."
                )
            ),
            HumanMessage(
                content=(
                    f"The maximum number of axes is {MAX_AXES}. "
                    "Here are the current axes:\n"
                    f"{axis_lines}"
                )
            ),
        ]
        log_llm_prompt_artifact("llm-prompt-axis-merge-suggestion", messages)
        response = llm.invoke(messages)
        suggestion = _normalize_axis_suggestion(getattr(response, "content", ""))
        if span is not None:
            span.set_outputs({"suggestion": suggestion})
    logger.info("LLM call end: suggest_axis_merge_or_substitution")
    return suggestion


def _normalize_axis_suggestion(content: object) -> str:
    """Normalize the axis suggestion into the expected user-facing format."""

    if not isinstance(content, str):
        return AXIS_MERGE_FALLBACK

    suggestion = content.strip()
    if not suggestion:
        return AXIS_MERGE_FALLBACK
    if suggestion.startswith("I can suggest"):
        return suggestion
    return f"I can suggest... {suggestion}"


def _assess_axis_clarification_needs(
    axes: list[Axis],
    axis_issue_message: str | None,
    prompt: str,
    clarification: str | None,
) -> ClarificationAssessment | None:
    """Use the LLM to decide whether the axes or their descriptions need clarification."""

    llm = create_azure_chat_model()
    if llm is None:
        return _fallback_axis_assessment(axis_issue_message=axis_issue_message)

    structured_llm = llm.with_structured_output(ClarificationAssessment)
    axis_lines = "\n".join(
        f"- {axis.name or '(unnamed axis)'}: {axis.description or '(no description)'}"
        for axis in axes
    )
    existing_axis_issue = axis_issue_message or "None."

    logger.info(
        "LLM call start: assess_axis_clarification_needs axis_count=%s axis_issue_present=%s",
        len(axes),
        bool(axis_issue_message),
    )
    with traced_operation(
        "llm.assess_axis_clarification_needs",
        span_type=SpanType.GUARDRAIL,
        inputs={
            "axis_count": len(axes),
            "axis_issue_present": bool(axis_issue_message),
            "clarification_present": bool((clarification or "").strip()),
        },
    ) as span:
        messages = [
            SystemMessage(
                content=(
                    "All inputs in this conversation are related to professional "
                    "employment and work preferences. Interpret them only in that "
                    "context. "
                    "You validate company-fit matching axes only. Decide whether the "
                    "axes need clarification, merging, substitution, removal, or more "
                    "specific descriptions. Axes should reflect the user's stated "
                    "priorities from the prompt, not a decomposition of CV details into "
                    "extra axes. Apply this rule strictly: only requirements that can "
                    "be validated through official company information or other "
                    "authoritative company facts belong to search criteria. Absolute "
                    "or near-absolute official company facts such as location, size, "
                    "domain, ownership model, or funding stage belong to search "
                    "criteria. Everything else belongs in axes. Axes are only the "
                    "parts of the user's request that require interpretation, "
                    "investigation, assumptions, or probabilistic judgment about fit. "
                    "If a prompt contains both official company filters and "
                    "non-official preferences, treat that as a normal mixed prompt, "
                    "not as missing axes. Non-official preferences include skill or "
                    "stack match, work-arrangement preferences, role-fit judgments, "
                    "compensation fit, culture or team fit, transition friendliness, "
                    "and similar fit questions that are not official company facts. "
                    "Never ask the user to add axes for company-side search filters. "
                    "Broad but intentional axes are valid if they are clear enough to "
                    "score using CV evidence and company or role data. "
                    "Do not ask the user to split a broad axis into smaller axes unless "
                    "the axis is truly ambiguous or internally contradictory. "
                    "If the prompt includes both search filters and softer fit "
                    "priorities, only the softer fit priorities belong in axes. "
                    "If there is an existing axis issue message, preserve its meaning in "
                    "the final user_message. Do not ask about CV details."
                )
            ),
            HumanMessage(
                content=(
                    f"User prompt:\n{prompt.strip()}\n\n"
                    f"Latest user clarification:\n{(clarification or '').strip() or 'None'}\n\n"
                    f"Axes:\n{axis_lines}\n\n"
                    f"Existing axis issue message:\n{existing_axis_issue}"
                )
            ),
        ]
        log_llm_prompt_artifact("llm-prompt-assess-axis-clarification", messages)
        result = structured_llm.invoke(messages)
        if span is not None:
            span.set_outputs(result.model_dump())
    logger.info(
        "LLM call end: assess_axis_clarification_needs needs_clarification=%s",
        result.needs_clarification,
    )
    return result


def _assess_cv_sufficiency(
    axes: list[Axis],
    simplified_cv_text: str,
    prompt: str,
    clarification: str | None,
) -> ClarificationAssessment | None:
    """Use the LLM to decide whether the simplified CV lacks needed detail."""

    llm = create_azure_chat_model()
    if llm is None:
        return None

    structured_llm = llm.with_structured_output(ClarificationAssessment)
    axis_lines = "\n".join(
        f"- {axis.name or '(unnamed axis)'}: {axis.description or '(no description)'}"
        for axis in axes
    )

    logger.info(
        "LLM call start: assess_cv_sufficiency axis_count=%s simplified_cv_chars=%s",
        len(axes),
        len(simplified_cv_text.strip()),
    )
    with traced_operation(
        "llm.assess_cv_sufficiency",
        span_type=SpanType.GUARDRAIL,
        inputs={
            "axis_count": len(axes),
            "simplified_cv_chars": len(simplified_cv_text.strip()),
            "clarification_present": bool((clarification or "").strip()),
        },
    ) as span:
        messages = [
            SystemMessage(
                content=(
                    "All inputs in this conversation are related to professional "
                    "employment and work preferences. Interpret them only in that "
                    "context. "
                    "You evaluate only whether the simplified CV contains enough detail "
                    "to assess company fit against the user's prompt and axes. Decide "
                    "whether more CV information is needed. Use a best-effort reading "
                    "of the CV and the latest user clarification before asking for more. "
                    "Only ask for additional CV details when they are strictly needed "
                    "to evaluate one or more current axes. Do not ask generic job "
                    "application or recruiting questions. Do not ask about visa "
                    "status, work authorization, location, relocation, or preferred "
                    "work arrangement unless that topic is explicitly part of the "
                    "user's prompt or one of the axes being evaluated. "
                    "If the user says the information is already in the CV, do not ask "
                    "them to restate the same CV content unless the information is truly "
                    "absent or impossible to infer even approximately. If clarification "
                    "is needed, ask only direct follow-up questions about missing CV "
                    "details. Do not comment on axis quality or ask the user to revise axes."
                )
            ),
            HumanMessage(
                content=(
                    f"User prompt:\n{prompt.strip()}\n\n"
                    f"Latest user clarification:\n{(clarification or '').strip() or 'None'}\n\n"
                    f"Simplified CV:\n{simplified_cv_text.strip()}\n\n"
                    f"Axes:\n{axis_lines}"
                )
            ),
        ]
        log_llm_prompt_artifact("llm-prompt-assess-cv-sufficiency", messages)
        result = structured_llm.invoke(messages)
        if span is not None:
            span.set_outputs(result.model_dump())
    logger.info(
        "LLM call end: assess_cv_sufficiency needs_clarification=%s",
        result.needs_clarification,
    )
    return result


def _fallback_axis_assessment(
    axis_issue_message: str | None,
) -> ClarificationAssessment | None:
    """Return deterministic axis clarification guidance when the LLM is unavailable."""

    if axis_issue_message:
        logger.info("Using fallback axis assessment because LLM is unavailable.")
        return ClarificationAssessment(
            needs_clarification=True,
            user_message=axis_issue_message,
        )
    return None
