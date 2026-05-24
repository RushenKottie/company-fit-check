"""Patchable deterministic stub implementations for eval cases."""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from mlflow.entities.span import SpanType

from models.state import (
    Axis,
    CompanyCandidate,
    CompanyScore,
    CompanySearchCriteria,
    UserInputInterpretation,
)
from services.mlflow_tracking import (
    log_llm_prompt_artifact,
    traced_operation,
)


class FakeGuardrail4xxError(RuntimeError):
    """Deterministic fake LLM guardrail error with a 4xx status code."""

    def __init__(self, message: str = "Guardrail blocked the request with status 400.") -> None:
        super().__init__(message)
        self.status_code = 400


def stub_simplify_masked_cv(masked_cv_text: str) -> str:
    """Deterministic replacement for the LLM-based CV simplifier."""

    simplified = (
        "Professional Profile\n\n"
        "Seniority: Mid-level product professional.\n"
        "Core Expertise: Fintech, operations, and stakeholder coordination.\n"
        "Technical Depth: Uses masked CV facts only.\n"
        "Confidence Notes: Deterministic eval stub."
    )
    with traced_operation(
        "llm.simplify_cv",
        span_type=SpanType.LLM,
        inputs={"masked_cv_chars": len(masked_cv_text)},
    ) as span:
        messages = [HumanMessage(content=f"Masked CV text:\n\n{masked_cv_text}")]
        log_llm_prompt_artifact("stub-llm-prompt-simplify-cv", messages)
        if span is not None:
            span.set_outputs({"simplified_cv_text": simplified})
    return simplified


def stub_interpret_user_input(
    prompt: str,
    simplified_cv_text: str,
    clarification: str | None = None,
    previous_axes: list[Axis] | None = None,
) -> UserInputInterpretation:
    """Deterministic replacement for user-input interpretation."""

    interpretation = UserInputInterpretation(
        company_search_criteria=CompanySearchCriteria(
            industries=["fintech"],
            roles=["product manager"],
        ),
        axes=[
            Axis(
                name="career-switch openness",
                description="How willing the company is to value transferable experience.",
            )
        ],
    )
    with traced_operation(
        "llm.interpret_user_input",
        span_type=SpanType.LLM,
        inputs={"prompt_chars": len(prompt), "simplified_cv_chars": len(simplified_cv_text)},
    ) as span:
        messages = [
            HumanMessage(
                content=(
                    f"User prompt:\n{prompt}\n\n"
                    f"Simplified CV:\n{simplified_cv_text}\n\n"
                    f"Clarification:\n{clarification or 'None'}"
                )
            )
        ]
        log_llm_prompt_artifact("stub-llm-prompt-interpret-user-input", messages)
        if span is not None:
            span.set_outputs(
                {
                    "axes": [axis.model_dump() for axis in interpretation.axes],
                    "company_search_criteria": interpretation.company_search_criteria.model_dump(),
                }
            )
    return interpretation


def stub_interpret_user_input_returns_only_search_criteria(
    prompt: str,
    simplified_cv_text: str,
    clarification: str | None = None,
    previous_axes: list[Axis] | None = None,
) -> UserInputInterpretation:
    """Return only deterministic search criteria so fallback-axis behavior can be tested."""

    return UserInputInterpretation(
        company_search_criteria=CompanySearchCriteria(
            locations=["Cyprus", "Paphos"],
            roles=["backend qa"],
            work_modes=["remote", "hybrid"],
            must_have=["java"],
        ),
        axes=[],
    )


def stub_interpret_user_input_requires_safe_rephrase(
    prompt: str,
    simplified_cv_text: str,
    clarification: str | None = None,
    previous_axes: list[Axis] | None = None,
) -> UserInputInterpretation:
    """Raise a guardrail error until the prompt is replaced with a safe reformulation."""

    if "safe reformulation" not in prompt.lower():
        raise FakeGuardrail4xxError(
            "Azure content filter blocked the request with status code 400."
        )
    return stub_interpret_user_input(
        prompt=prompt,
        simplified_cv_text=simplified_cv_text,
        clarification=clarification,
        previous_axes=previous_axes,
    )


def stub_interpret_user_input_requires_safe_prompt_without_clarification(
    prompt: str,
    simplified_cv_text: str,
    clarification: str | None = None,
    previous_axes: list[Axis] | None = None,
) -> UserInputInterpretation:
    """Succeed only when a guardrail rephrase replaces prior clarification context."""

    if not previous_axes:
        return stub_interpret_user_input(
            prompt=prompt,
            simplified_cv_text=simplified_cv_text,
            clarification=clarification,
            previous_axes=previous_axes,
        )
    if "safe reformulation" not in prompt.lower() or clarification is not None:
        raise FakeGuardrail4xxError(
            "Azure content filter blocked the request with status code 400."
        )
    return stub_interpret_user_input(
        prompt=prompt,
        simplified_cv_text=simplified_cv_text,
        clarification=clarification,
        previous_axes=previous_axes,
    )


def stub_validate_user_input_interpretation(
    axes: list[Axis],
    simplified_cv_text: str,
    prompt: str,
    clarification: str | None = None,
) -> tuple[bool, str | None]:
    """Deterministic replacement for user-input interpretation validation."""

    return True, None


def stub_validate_user_input_interpretation_needs_clarification(
    axes: list[Axis],
    simplified_cv_text: str,
    prompt: str,
    clarification: str | None = None,
) -> tuple[bool, str | None]:
    """Deterministic replacement that forces clarification."""

    return False, "Please clarify the intended meaning of your axes."


def stub_validate_user_input_interpretation_needs_clarification_once(
    axes: list[Axis],
    simplified_cv_text: str,
    prompt: str,
    clarification: str | None = None,
) -> tuple[bool, str | None]:
    """Ask for clarification once, then accept the reformulated prompt."""

    if clarification or "safe reformulation" in prompt.lower():
        return True, None
    return False, "Please clarify the intended meaning of your axes."


def stub_discover_companies(
    company_search_criteria: CompanySearchCriteria,
    limit: int = 500,
):
    """Deterministic replacement for company discovery."""

    companies = [
        CompanyCandidate(
            name="Fintech Labs",
            website_or_linkedin="https://fintechlabs.example.com",
            industry="fintech",
            company_size="201-500",
            discovery_reason="Matches the deterministic fintech criteria.",
            confidence=0.9,
        ),
        CompanyCandidate(
            name="Payments Studio",
            website_or_linkedin="https://paymentsstudio.example.com",
            industry="fintech",
            company_size="51-200",
            discovery_reason="Matches the deterministic product-manager criteria.",
            confidence=0.8,
        ),
    ]
    with traced_operation(
        "llm.discover_companies",
        span_type=SpanType.RETRIEVER,
        inputs={"limit": limit},
    ) as span:
        messages = [
            HumanMessage(
                content=f"Company search criteria:\n{company_search_criteria.model_dump_json(indent=2)}"
            )
        ]
        log_llm_prompt_artifact("stub-llm-prompt-discover-companies", messages)
        if span is not None:
            span.set_outputs(
                {
                    "company_count": len(companies),
                    "companies": [company.model_dump() for company in companies],
                }
            )
    return companies


def stub_score_companies(
    companies: list[CompanyCandidate],
    simplified_cv_text: str,
    axes: list[Axis],
) -> list[CompanyScore]:
    """Deterministic replacement for company scoring."""

    scores = [
        CompanyScore(
            company_name=company.name,
            axis_scores=[
                {
                    "axis": axis.name,
                    "percentage": 72.0 + index,
                }
                for index, axis in enumerate(axes)
            ],
            overall_score=72.0,
        )
        for company in companies
    ]
    with traced_operation(
        "llm.score_company_batch",
        span_type=SpanType.EVALUATOR,
        inputs={"company_count": len(companies), "axes_count": len(axes)},
    ) as span:
        messages = [
            HumanMessage(
                content=(
                    f"Simplified CV:\n{simplified_cv_text}\n\n"
                    f"Companies:\n{[company.name for company in companies]}"
                )
            )
        ]
        log_llm_prompt_artifact("stub-llm-prompt-score-companies", messages)
        if span is not None:
            span.set_outputs(
                {
                    "company_scores": [score.model_dump() for score in scores],
                }
            )
    return scores


def raise_runtime_error(*args, **kwargs):
    """Deterministic replacement that raises a runtime error."""

    raise RuntimeError("Injected deterministic failure.")


def raise_guardrail_4xx_error(*args, **kwargs):
    """Deterministic replacement that raises a fake guardrail 4xx error."""

    raise FakeGuardrail4xxError(
        "Azure content filter blocked the request with status code 400."
    )


def raise_guardrail_message_only_error(*args, **kwargs):
    """Deterministic replacement that raises a guardrail-like error without status metadata."""

    raise RuntimeError(
        "Azure content filter blocked the request because of safety policy."
    )


STUB_REGISTRY = {
    "stub_simplify_masked_cv": stub_simplify_masked_cv,
    "stub_interpret_user_input": stub_interpret_user_input,
    "stub_interpret_user_input_returns_only_search_criteria": stub_interpret_user_input_returns_only_search_criteria,
    "stub_interpret_user_input_requires_safe_rephrase": stub_interpret_user_input_requires_safe_rephrase,
    "stub_interpret_user_input_requires_safe_prompt_without_clarification": stub_interpret_user_input_requires_safe_prompt_without_clarification,
    "stub_validate_user_input_interpretation": stub_validate_user_input_interpretation,
    "stub_validate_user_input_interpretation_needs_clarification": stub_validate_user_input_interpretation_needs_clarification,
    "stub_validate_user_input_interpretation_needs_clarification_once": stub_validate_user_input_interpretation_needs_clarification_once,
    "stub_discover_companies": stub_discover_companies,
    "stub_score_companies": stub_score_companies,
    "raise_runtime_error": raise_runtime_error,
    "raise_guardrail_4xx_error": raise_guardrail_4xx_error,
    "raise_guardrail_message_only_error": raise_guardrail_message_only_error,
}
