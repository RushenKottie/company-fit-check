"""Azure-hosted LLM client configuration."""

from functools import lru_cache
import importlib
import re
from types import SimpleNamespace

from langchain_core.messages import SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI

from config import (
    get_azure_openai_settings,
    get_llm_judge_azure_openai_settings,
    get_user_simulator_azure_openai_settings,
    get_user_simulator_foundry_settings,
)
from logging_utils import get_logger

logger = get_logger(__name__)
SUPPORT_EMAIL = "support@example.com"
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


class AnthropicFoundryChatAdapter:
    """Small adapter that exposes Anthropic Foundry through ``invoke(messages)``."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float | None = None,
        max_tokens: int,
    ) -> None:
        anthropic_module = importlib.import_module("anthropic")
        self._client = anthropic_module.AnthropicFoundry(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def invoke(self, messages: list[object]) -> object:
        """Translate chat-style messages into the Anthropic Messages API."""

        system_parts: list[str] = []
        anthropic_messages: list[dict[str, str]] = []

        for message in messages:
            content = _normalize_message_content(message)
            if not content:
                continue
            if isinstance(message, SystemMessage):
                system_parts.append(content)
                continue
            role = "assistant" if _message_role(message) == "ai" else "user"
            anthropic_messages.append({"role": role, "content": content})

        request_kwargs = {
            "model": self._model,
            "system": "\n\n".join(system_parts) if system_parts else None,
            "messages": anthropic_messages,
            "max_tokens": self._max_tokens,
        }
        if self._temperature is not None:
            request_kwargs["temperature"] = self._temperature

        response = self._client.messages.create(
            **request_kwargs,
        )
        return SimpleNamespace(content=response.content)


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


@lru_cache(maxsize=1)
def create_user_simulator_azure_chat_model() -> BaseChatModel | None:
    """Return a cached AzureChatOpenAI client for the user simulator."""

    settings = get_user_simulator_azure_openai_settings()
    if not settings.is_configured:
        logger.warning(
            "User simulator Azure chat model requested but required settings are incomplete."
        )
        return None

    logger.info(
        "Creating user simulator Azure chat model deployment=%s api_version=%s max_tokens=%s temperature=%s",
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


@lru_cache(maxsize=1)
def create_user_simulator_chat_model() -> object | None:
    """Return the configured chat model for the user simulator."""

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
    return AnthropicFoundryChatAdapter(
        api_key=foundry_settings.api_key or "",
        base_url=foundry_settings.endpoint or "",
        model=foundry_settings.model or "",
        temperature=foundry_settings.temperature,
        max_tokens=foundry_settings.max_tokens,
    )


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


def build_generic_llm_failure_message() -> str:
    """Return the generic user-facing LLM failure message."""

    return (
        "Sorry, something went wrong. Please try again later or contact "
        f"{SUPPORT_EMAIL}."
    )


def build_restart_due_to_guardrail_message(stage: str) -> str:
    """Return the user-facing restart message for terminal guardrail failures."""

    return (
        "Sorry, I couldn't continue because the model blocked this request during "
        f"{stage}. Please start a new session with another prompt or CV."
    )


def build_rephrase_due_to_guardrail_message() -> str:
    """Return the user-facing rephrase message for recoverable guardrail failures."""

    return (
        "Please reformulate your message and try again so I can continue the workflow."
    )


def build_rephrase_retry_exhausted_message() -> str:
    """Return the user-facing message for repeated reformulation failures."""

    return (
        "Sorry, I still couldn't continue after repeated reformulation attempts. "
        "Please start a new session with another prompt or CV."
    )


def build_cv_cleanup_due_to_guardrail_message() -> str:
    """Return the user-facing message for CV-content guardrail failures."""

    return (
        "Your CV appears to contain trigger words or sensitive topics. Please remove "
        "topics like fraud, scam, abuse, violence, harassment, and similar content, "
        "then start a new session with the cleaned CV."
    )


def is_guardrail_rephrase_message(message: str | None) -> bool:
    """Return whether the clarification message is the fixed guardrail rephrase prompt."""

    return (message or "").strip() == build_rephrase_due_to_guardrail_message()


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


def _normalize_message_content(message: object) -> str:
    """Return one plain-text message content string."""

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif hasattr(item, "text") and isinstance(item.text, str):
                parts.append(item.text)
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    return str(content).strip()


def _message_role(message: object) -> str:
    """Return a normalized chat role name for one message object."""

    return str(getattr(message, "type", message.__class__.__name__)).lower()
