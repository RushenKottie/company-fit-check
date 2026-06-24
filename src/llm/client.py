"""Azure-hosted LLM client configuration."""

from functools import lru_cache
import re

from anthropic import AnthropicFoundry
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI

from config import (
    get_azure_openai_settings,
    get_llm_judge_azure_openai_settings,
    get_user_simulator_foundry_settings,
)
from logging_utils import get_logger

logger = get_logger(__name__)
_GUARDRAIL_KEYWORDS = (
    "guardrail",
    "content filter",
    "content_filter",
    "content policy",
    "safety policy",
    "responsible ai policy",
    "jailbreak",
    "violence",
    "harassment",
    "sexual",
    "hate",
    "self-harm",
    "blocked",
)


@lru_cache(maxsize=1)
def create_azure_chat_model() -> BaseChatModel | None:
    """Return a cached AzureChatOpenAI client when configuration is complete."""

    settings = get_azure_openai_settings()
    if not settings.is_configured:
        logger.warning("Azure chat model requested but required settings are incomplete.")
        return None

    logger.info(
        "Creating Azure chat model client deployment=%s api_version=%s max_tokens=%s temperature=%s",
        settings.deployment,
        settings.api_version,
        settings.max_tokens,
        settings.temperature,
    )
    return AzureChatOpenAI(
        api_key=settings.api_key,
        azure_endpoint=settings.endpoint,
        azure_deployment=settings.deployment,
        api_version=settings.api_version,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


def generate_user_simulator_reply(prompt: str) -> str | None:
    """Return one plain-text user simulator reply from Anthropic Foundry."""

    foundry_settings = get_user_simulator_foundry_settings()
    if not foundry_settings.is_configured:
        logger.warning(
            "User simulator Anthropic Foundry client requested but required settings are not configured."
        )
        return None

    logger.info(
        "Creating user simulator Anthropic Foundry client model=%s endpoint=%s max_tokens=%s temperature=%s",
        foundry_settings.model,
        foundry_settings.endpoint,
        foundry_settings.max_tokens,
        foundry_settings.temperature,
    )
    client = AnthropicFoundry(
        api_key=foundry_settings.api_key or "",
        base_url=foundry_settings.endpoint or "",
    )

    response = client.messages.create(
        model=foundry_settings.model or "",
        temperature=foundry_settings.temperature,
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=foundry_settings.max_tokens,
    )
    if not response.content:
        return ""
    return response.content[0].text


@lru_cache(maxsize=1)
def create_llm_judge_chat_model() -> BaseChatModel | None:
    """Return a cached AzureChatOpenAI client for the regression judge."""

    settings = get_llm_judge_azure_openai_settings()
    if not settings.is_configured:
        logger.warning("LLM judge chat model requested but required settings are incomplete.")
        return None

    logger.info(
        "Creating LLM judge Azure chat model deployment=%s api_version=%s max_tokens=%s temperature=%s",
        settings.deployment,
        settings.api_version,
        settings.max_tokens,
        settings.temperature,
    )
    return AzureChatOpenAI(
        api_key=settings.api_key,
        azure_endpoint=settings.endpoint,
        azure_deployment=settings.deployment,
        api_version=settings.api_version,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


def is_guardrail_4xx_error(exc: Exception) -> bool:
    """Return whether the exception looks like an LLM guardrail/content-filter 4xx."""

    status_code = _extract_status_code(exc)
    if status_code is not None:
        return 400 <= status_code < 500

    message = str(exc).lower()
    return any(keyword in message for keyword in _GUARDRAIL_KEYWORDS)


def _extract_status_code(exc: Exception) -> int | None:
    """Extract an HTTP-style status code from nested exception objects when available."""

    for candidate in (
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
        getattr(getattr(exc, "__cause__", None), "status_code", None),
    ):
        if isinstance(candidate, int):
            return candidate
        if isinstance(candidate, str) and candidate.isdigit():
            return int(candidate)

    message = str(exc)
    match = re.search(r"\b([45]\d{2})\b", message)
    if match:
        return int(match.group(1))
    return None
