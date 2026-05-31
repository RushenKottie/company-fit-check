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
class UserSimulatorFoundrySettings:
    """Resolved Anthropic Foundry settings for the user simulator."""

    endpoint: str | None
    api_key: str | None
    model: str | None
    temperature: float | None
    max_tokens: int

    @property
    def is_configured(self) -> bool:
        """Return whether the minimum Foundry config is present."""

        return bool(self.endpoint and self.api_key and self.model)


@dataclass(frozen=True)
class LlmJudgeSettings:
    """Resolved Azure OpenAI settings for the regression judge."""

    deployment: str | None
    api_version: str
    temperature: float
    max_tokens: int

    @property
    def is_configured(self) -> bool:
        """Return whether the minimum judge deployment config is present."""

        return bool(self.deployment)


@dataclass(frozen=True)
class MlflowSettings:
    """Resolved MLflow and Azure Blob settings for the app."""

    tracking_uri: str
    experiment_name: str
    regression_experiment_name: str
    mutation_experiment_name: str
    artifact_root: str | None
    azure_storage_connection_string: str | None

    @property
    def is_configured(self) -> bool:
        """Return whether the MLflow + Blob persistence config is complete."""

        if not self.tracking_uri:
            return False
        if not self.artifact_root:
            return True
        if self.artifact_root.startswith("wasbs://"):
            return bool(self.azure_storage_connection_string)
        return True


def _project_root() -> Path:
    """Return the repository root based on this module location."""

    return Path(__file__).resolve().parents[1]


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
def get_user_simulator_azure_openai_settings() -> AzureOpenAISettings:
    """Load Azure OpenAI settings for the user simulator from environment values."""

    values = _load_env_values()

    deployment = values.get("USER_SIMULATOR_AZURE_OPENAI_DEPLOYMENT")
    api_version = (
        values.get("USER_SIMULATOR_AZURE_OPENAI_API_VERSION")
        or values["AZURE_OPENAI_API_VERSION"]
    )
    temperature = float(
        values.get("USER_SIMULATOR_AZURE_OPENAI_TEMPERATURE")
        or values.get("AZURE_OPENAI_TEMPERATURE")
        or "0"
    )
    max_tokens = int(
        values.get("USER_SIMULATOR_AZURE_OPENAI_MAX_TOKENS")
        or values.get("AZURE_OPENAI_MAX_TOKENS")
        or "5000"
    )

    return AzureOpenAISettings(
        deployment=deployment,
        api_version=api_version,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@lru_cache(maxsize=1)
def get_user_simulator_foundry_settings() -> UserSimulatorFoundrySettings:
    """Load Anthropic Foundry settings for the user simulator from environment values."""

    values = _load_env_values()
    temperature_value = values.get("USER_SIMULATOR_FOUNDRY_TEMPERATURE")
    max_tokens = int(
        values.get("USER_SIMULATOR_FOUNDRY_MAX_TOKENS")
        or "5000"
    )

    return UserSimulatorFoundrySettings(
        endpoint=values.get("USER_SIMULATOR_FOUNDRY_ENDPOINT"),
        api_key=values.get("USER_SIMULATOR_FOUNDRY_API_KEY"),
        model=values.get("USER_SIMULATOR_FOUNDRY_MODEL"),
        temperature=float(temperature_value) if temperature_value else None,
        max_tokens=max_tokens,
    )


@lru_cache(maxsize=1)
def get_llm_judge_azure_openai_settings() -> LlmJudgeSettings:
    """Load Azure OpenAI settings for the regression judge from environment values."""

    values = _load_env_values()

    deployment = (
        values.get("LLM_JUDGE_AZURE_OPENAI_DEPLOYMENT")
        or values.get("AZURE_OPENAI_DEPLOYMENT")
    )
    api_version = (
        values.get("LLM_JUDGE_AZURE_OPENAI_API_VERSION")
        or values.get("AZURE_OPENAI_API_VERSION")
    )
    temperature = float(
        values.get("LLM_JUDGE_AZURE_OPENAI_TEMPERATURE")
        or values.get("AZURE_OPENAI_TEMPERATURE")
        or "0"
    )
    max_tokens = int(
        values.get("LLM_JUDGE_AZURE_OPENAI_MAX_TOKENS")
        or values.get("AZURE_OPENAI_MAX_TOKENS")
        or "5000"
    )

    return LlmJudgeSettings(
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
    regression_experiment_name = (
        values.get("MLFLOW_REGRESSION_EXPERIMENT_NAME")
        or "company-fit-check-llm-regression"
    )
    mutation_experiment_name = (
        values.get("MLFLOW_MUTATION_EXPERIMENT_NAME")
        or "company-fit-check-llm-mutation"
    )
    artifact_root = values.get("MLFLOW_ARTIFACT_ROOT") or None
    azure_storage_connection_string = (
        values.get("AZURE_STORAGE_CONNECTION_STRING") or None
    )

    return MlflowSettings(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        regression_experiment_name=regression_experiment_name,
        mutation_experiment_name=mutation_experiment_name,
        artifact_root=artifact_root,
        azure_storage_connection_string=azure_storage_connection_string,
    )
