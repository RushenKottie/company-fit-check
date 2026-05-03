"""Azure-hosted LLM client configuration."""

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI

from company_fit_check.config import get_azure_openai_settings
from company_fit_check.logging_utils import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def create_azure_chat_model() -> BaseChatModel | None:
    """Return a cached AzureChatOpenAI client when configuration is complete."""

    settings = get_azure_openai_settings()
    if not settings.is_configured:
        logger.warning("Azure chat model requested but deployment is not configured.")
        return None

    logger.info(
        "Creating Azure chat model client deployment=%s api_version=%s max_tokens=%s temperature=%s",
        settings.deployment,
        settings.api_version,
        settings.max_tokens,
        settings.temperature,
    )
    return AzureChatOpenAI(
        azure_deployment=settings.deployment,
        api_version=settings.api_version,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
