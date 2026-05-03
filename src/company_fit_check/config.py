"""Minimal runtime configuration helpers."""

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

from dotenv import dotenv_values, find_dotenv, load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AzureOpenAISettings:
    """Resolved Azure OpenAI settings for the app."""

    deployment: str | None
    api_version: str
    temperature: float
    max_tokens: int

    @property
    def is_configured(self) -> bool:
        """Return whether the minimum Azure deployment config is present."""

        return bool(self.deployment)


@dataclass(frozen=True)
class MlflowSettings:
    """Resolved MLflow and Azure Blob settings for the app."""

    tracking_uri: str
    experiment_name: str
    artifact_root: str | None
    azure_storage_connection_string: str | None

    @property
    def is_configured(self) -> bool:
        """Return whether the MLflow + Blob persistence config is complete."""

        return bool(self.artifact_root and self.azure_storage_connection_string)


def _project_root() -> Path:
    """Return the repository root based on this module location."""

    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _load_env_values() -> dict[str, str]:
    """Load .env values once and merge them with the current process environment."""

    env_path = find_dotenv(usecwd=True)
    values = dotenv_values(env_path) if env_path else {}
    merged = {key: value for key, value in values.items() if value is not None}
    for key, value in os.environ.items():
        if value:
            merged[key] = value
    return merged


@lru_cache(maxsize=1)
def get_azure_openai_settings() -> AzureOpenAISettings:
    """Load Azure OpenAI settings once from the local environment file."""

    values = _load_env_values()

    deployment = values["AZURE_OPENAI_DEPLOYMENT"]
    api_version = values["AZURE_OPENAI_API_VERSION"]
    temperature = float(values.get("AZURE_OPENAI_TEMPERATURE") or "0")
    max_tokens = int(values.get("AZURE_OPENAI_MAX_TOKENS") or "5000")

    return AzureOpenAISettings(
        deployment=deployment,
        api_version=api_version,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@lru_cache(maxsize=1)
def get_mlflow_settings() -> MlflowSettings:
    """Load MLflow and Azure Blob settings from environment values."""

    values = _load_env_values()
    default_tracking_path = (_project_root() / ".mlruns").resolve()
    tracking_uri = values.get("MLFLOW_TRACKING_URI") or default_tracking_path.as_uri()
    experiment_name = values.get("MLFLOW_EXPERIMENT_NAME") or "company-fit-check"
    artifact_root = values.get("MLFLOW_ARTIFACT_ROOT") or None
    azure_storage_connection_string = (
        values.get("AZURE_STORAGE_CONNECTION_STRING") or None
    )

    return MlflowSettings(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_root=artifact_root,
        azure_storage_connection_string=azure_storage_connection_string,
    )
