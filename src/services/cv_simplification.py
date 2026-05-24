"""CV simplification over already masked CV text."""

from langchain_core.messages import HumanMessage, SystemMessage
from mlflow.entities.span import SpanType

from llm.client import create_azure_chat_model
from logging_utils import get_logger
from services.mlflow_tracking import log_llm_prompt_artifact, traced_operation

logger = get_logger(__name__)


def simplify_masked_cv(masked_cv_text: str) -> str:
    """Simplify masked CV text with Azure OpenAI."""

    if not masked_cv_text:
        raise ValueError("Masked CV text is empty.")

    llm = create_azure_chat_model()
    if llm is None:
        raise RuntimeError("Azure OpenAI is not configured for CV simplification.")

    logger.info("LLM call start: simplify_masked_cv input_chars=%s", len(masked_cv_text))
    with traced_operation(
        "llm.simplify_cv",
        span_type=SpanType.LLM,
        inputs={"masked_cv_chars": len(masked_cv_text)},
    ) as span:
        messages = [
            SystemMessage(
                content=(
                    "You transform masked CV text into a concise professional "
                    "profile for downstream company matching. This is not a generic "
                    "summary. Your job is to convert experience descriptions into "
                    "stable professional signals that help evaluate company and role "
                    "fit. Return plain text only. Use only information explicitly "
                    "present in the CV. Do not invent facts. Do not speculate beyond "
                    "the evidence. You may infer patterns only when they are clearly "
                    "supported by multiple signals in the CV. If evidence is weak or "
                    "ambiguous, mark that conclusion as low confidence. Do not repeat "
                    "the CV verbatim. Avoid marketing language, praise, and "
                    "subjective opinions. Preserve concrete technical, product, and "
                    "business context when present, including product or platform "
                    "names, system types, and domain context such as payments, ATM, "
                    "wallets, core banking, fraud, KYC/AML, mobile banking, B2B, "
                    "B2C, internal systems, or customer-facing platforms. Do not "
                    "collapse specific context into vague labels when the CV contains "
                    "useful detail. Keep each field short, about 1 to 3 sentences. "
                    "Preserve education, formal training, and certifications when "
                    "they materially affect domain fit, career-switch potential, "
                    "or credibility for the roles being matched. Do "
                    "not drop an engineering or science degree just because recent "
                    "work experience is in another area. Include certifications "
                    "only when they are explicitly present in the CV. "
                    "Do not use markdown bullets or JSON. Output readable structured "
                    "text with exactly these fields and no others:\n\n"
                    "Professional Profile\n\n"
                    "Seniority:\n"
                    "Core Expertise:\n"
                    "Engineering Scope:\n"
                    "Responsibility Signals:\n"
                    "Delivery Orientation:\n"
                    "Technical Depth:\n"
                    "Domain Context:\n"
                    "Education Signals:\n"
                    "Certifications or Formal Training:\n"
                    "Work Context:\n"
                    "Working Style Signals:\n"
                    "Career Direction Signals:\n"
                    "Constraints or Preferences:\n"
                    "Confidence Notes:"
                )
            ),
            HumanMessage(content=f"Masked CV text:\n\n{masked_cv_text}"),
        ]
        log_llm_prompt_artifact("llm-prompt-simplify-cv", messages)
        response = llm.invoke(messages)
        content = response.content
        simplified = content if isinstance(content, str) else str(content)
        simplified = simplified.strip()
        if not simplified:
            raise ValueError("LLM returned an empty simplified CV.")
        if span is not None:
            span.set_outputs({"simplified_cv_text": simplified})
    logger.info("LLM call end: simplify_masked_cv output_chars=%s", len(simplified))
    return simplified
