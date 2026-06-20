"""User-input interpretation into search criteria and axes."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from mlflow.entities.span import SpanType

from llm.client import create_azure_chat_model
from logging_utils import get_logger
from models.state import Axis, UserInputInterpretation
from infrastructure.mlflow_tracking import log_llm_prompt_artifact, traced_operation
from services.user_input_interpretation_validation import MAX_AXES

logger = get_logger(__name__)

INTERPRET_USER_INPUT_SYSTEM_PROMPT = (
    "All inputs in this conversation are related to professional "
    "employment and work preferences. Interpret them only in that "
    "context. "
    "Interpret a company-search prompt into search criteria and axes "
    "using the requested structured output schema. "
    "Use this separation rule strictly. "
    "company_search_criteria must contain only requirements that can "
    "be validated through official company information or other "
    "authoritative company facts. Examples include location, company "
    "size, industry or domain, ownership model, funding stage, and "
    "other explicit company attributes that can be verified directly. "
    "Concrete numerical, date, money, count, threshold, or range "
    "requirements must be stored in "
    "company_search_criteria.hard_filters. Hard filters are mandatory "
    "filters, not soft preferences, and must preserve the literal value "
    "the user provided. Fill minimum and maximum when the filter has "
    "numeric bounds. Examples include requested number of companies, "
    "founded after 2015, less than 500 employees, valuation above $1B, "
    "net worth over $10B, revenue under $100M, or funding between two "
    "amounts. Put requested output/result count filters in hard_filters "
    "with scope=result_set. Put per-company factual filters in "
    "hard_filters with scope=company_candidate. "
    "Axes must contain everything else from the user's prompt. "
    "If a requirement cannot be validated directly from official "
    "company information and instead needs interpretation, inference, "
    "investigation, or probabilistic judgment about fit, it belongs "
    "in axes. "
    "If the user's prompt mixes official company filters with "
    "non-official preferences, split them across both fields instead "
    "of forcing everything into company_search_criteria. "
    "Non-official preferences include role-fit judgments, skill or "
    "stack match, work-arrangement preferences, team or culture fit, "
    "career-transition friendliness, compensation fit, language "
    "environment, and other preferences that cannot be verified from "
    "official company facts alone. "
    "Every distinct non-official requirement from the user's prompt "
    "must appear in axes. Do not drop it, hide it inside search "
    "criteria, or leave it implicit. "
    "Do not merge distinct non-official requirements into one broad "
    "axis unless the user explicitly grouped them together as a "
    "single preference. If the user mentions multiple different "
    "non-official requirements, return multiple axes. "
    "If the prompt contains any non-official preference, axes must "
    "not be empty. "
    f"The maximum number of axes is {MAX_AXES}. If the user's "
    "preferences suggest more than that, merge overlapping or closely "
    "related non-official requirements into broader axes that still "
    "preserve the user's intended meaning. "
    "Axes must come from the user's prompt or clarification, not from "
    "decomposing the CV into separate dimensions. "
    "Search criteria are used only for company discovery. They are not "
    "matching dimensions and should not be restated as axes. "
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
    "traits or other special non-numerical constraints. Do not put "
    "hard numerical/date/money/count filters in undefined. "
    "Axis descriptions should explain the user's perspective when that "
    "perspective can be inferred."
)


def interpret_user_input(
    prompt: str,
    simplified_cv_text: str,
    clarification: str | None = None,
    previous_axes: list[Axis] | None = None,
) -> UserInputInterpretation:
    """Interpret the full user input and simplified CV into structured context."""

    if not prompt.strip():
        raise ValueError("User prompt is empty.")
    if not simplified_cv_text.strip():
        raise ValueError("Simplified CV text is empty.")

    llm = create_azure_chat_model()
    if llm is None:
        raise RuntimeError("Azure OpenAI is not configured for user-input interpretation.")

    structured_llm = llm.with_structured_output(UserInputInterpretation)
    logger.info(
        "LLM call start: interpret_user_input prompt_chars=%s simplified_cv_chars=%s clarification_present=%s previous_axes=%s",
        len(prompt.strip()),
        len(simplified_cv_text.strip()),
        bool(clarification and clarification.strip()),
        len(previous_axes or []),
    )
    with traced_operation(
        "llm.interpret_user_input",
        span_type=SpanType.LLM,
        inputs={
            "prompt_chars": len(prompt.strip()),
            "simplified_cv_chars": len(simplified_cv_text.strip()),
            "clarification_present": bool(clarification and clarification.strip()),
            "previous_axes_count": len(previous_axes or []),
        },
    ) as span:
        messages = _build_interpretation_messages(
            prompt=prompt,
            simplified_cv_text=simplified_cv_text,
            clarification=clarification,
            previous_axes=previous_axes,
        )
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
        "LLM call end: interpret_user_input axes=%s criteria_fields=%s",
        len(result.axes),
        sum(1 for value in result.company_search_criteria.model_dump().values() if value),
    )
    return result


def _build_interpretation_messages(
    prompt: str,
    simplified_cv_text: str,
    clarification: str | None,
    previous_axes: list[Axis] | None,
) -> list[object]:
    """Build the LLM messages for user-input interpretation."""

    return [
        SystemMessage(content=INTERPRET_USER_INPUT_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"User prompt:\n{prompt.strip()}\n\n"
                f"{_format_previous_axes_block(previous_axes)}"
                f"Simplified CV:\n{simplified_cv_text.strip()}\n"
                f"{_format_clarification_block(clarification)}"
            )
        ),
    ]


def _format_clarification_block(clarification: str | None) -> str:
    """Return the optional clarification block for the interpretation prompt."""

    if not clarification:
        return ""
    return f"\nUser clarification / additional context:\n{clarification.strip()}\n"


def _format_previous_axes_block(previous_axes: list[Axis] | None) -> str:
    """Return previous axes as prompt context when resuming after clarification."""

    if not previous_axes:
        return ""
    return (
        "\nExisting axes from the previous turn. Keep their meaning stable unless "
        "the user explicitly changed them:\n"
        + "\n".join(f"- {axis.name}: {axis.description}" for axis in previous_axes)
        + "\n"
    )
