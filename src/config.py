"""Runtime configuration helpers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
import os
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import dotenv_values, find_dotenv

if TYPE_CHECKING:
    from mlflow import MlflowClient


@dataclass(frozen=True)
class AzureOpenAISettings:
    """Resolved Azure OpenAI settings for the app."""

    api_key: str | None
    endpoint: str | None
    deployment: str | None
    api_version: str | None
    temperature: float
    max_tokens: int

    @property
    def is_configured(self) -> bool:
        """Return whether the Azure OpenAI client can be constructed."""

        return bool(
            self.api_key
            and self.endpoint
            and self.deployment
            and self.api_version
        )


@dataclass(frozen=True)
class UserSimulatorFoundrySettings:
    """Resolved Anthropic Foundry settings for the user simulator."""

    endpoint: str | None
    api_key: str | None
    model: str | None
    temperature: float
    max_tokens: int

    @property
    def is_configured(self) -> bool:
        """Return whether the minimum Foundry config is present."""

        return bool(self.endpoint and self.api_key and self.model)


@dataclass(frozen=True)
class LlmJudgeSettings:
    """Resolved Azure OpenAI settings for the regression judge."""

    api_key: str | None
    endpoint: str | None
    deployment: str | None
    api_version: str | None
    temperature: float
    max_tokens: int

    @property
    def is_configured(self) -> bool:
        """Return whether the judge Azure OpenAI client can be constructed."""

        return bool(
            self.api_key
            and self.endpoint
            and self.deployment
            and self.api_version
        )


class MlflowTrackingMode(StrEnum):
    """Supported MLflow tracking backends."""

    LOCAL = "local"
    AZUREML = "azureml"
    REMOTE = "remote"


@dataclass(frozen=True)
class MlflowSettings(ABC):
    """Resolved MLflow settings shared by all tracking backends."""

    tracking_uri: str
    experiment_name: str

    @property
    def is_configured(self) -> bool:
        """Return whether this validated settings object can be used."""

        return True

    @abstractmethod
    def configure_environment(self) -> None:
        """Prepare process environment required by the tracking backend."""

    @abstractmethod
    def create_experiment(self, client: "MlflowClient", experiment_name: str) -> str:
        """Create an experiment using backend-specific artifact behavior."""


@dataclass(frozen=True)
class LocalMlflowSettings(MlflowSettings):
    """Local MLflow SQL tracking with external artifact storage."""

    artifact_root: str
    azure_storage_connection_string: str | None

    def configure_environment(self) -> None:
        """Expose Azure Blob credentials for local MLflow artifact writes/reads."""

        if self.azure_storage_connection_string:
            os.environ["AZURE_STORAGE_CONNECTION_STRING"] = (
                self.azure_storage_connection_string
            )

    def create_experiment(self, client: "MlflowClient", experiment_name: str) -> str:
        """Create a local SQL-backed experiment with the configured artifact root."""

        return client.create_experiment(
            experiment_name,
            artifact_location=self.artifact_root,
        )


@dataclass(frozen=True)
class AzureMlflowSettings(MlflowSettings):
    """Azure ML / Foundry MLflow tracking settings."""

    def configure_environment(self) -> None:
        """Azure MLflow uses the configured azureml:// tracking URI."""

        return None

    def create_experiment(self, client: "MlflowClient", experiment_name: str) -> str:
        """Create an Azure MLflow experiment with Azure-managed artifacts."""

        return client.create_experiment(experiment_name)


@dataclass(frozen=True)
class RemoteMlflowSettings(MlflowSettings):
    """Remote MLflow tracking server settings."""

    def configure_environment(self) -> None:
        """Remote MLflow uses the configured HTTP tracking URI."""

        return None

    def create_experiment(self, client: "MlflowClient", experiment_name: str) -> str:
        """Create an experiment using the remote tracking server defaults."""

        return client.create_experiment(experiment_name)


@dataclass(frozen=True)
class LoggingSettings:
    """Resolved logging settings for the app."""

    level: str


def get_project_root() -> Path:
    """Return the repository root based on this module location."""

    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _load_env_values() -> dict[str, str]:
    """Resolve all environment-backed configuration values in one place."""

    env_path = find_dotenv(usecwd=True)
    file_values = dotenv_values(env_path) if env_path else {}
    merged = {
        key: stripped
        for key, value in file_values.items()
        if value is not None and (stripped := value.strip())
    }
    for key, value in os.environ.items():
        if stripped := value.strip():
            merged[key] = stripped
    return merged


def _env(values: dict[str, str], name: str, default: str | None = None) -> str | None:
    """Return one resolved env value from the centralized config map."""

    return values.get(name) or default


def _float_env(values: dict[str, str], name: str, default: str) -> float:
    """Return one float env value from the centralized config map."""

    return float(_env(values, name, default) or default)


def _int_env(values: dict[str, str], name: str, default: str) -> int:
    """Return one integer env value from the centralized config map."""

    return int(_env(values, name, default) or default)


@lru_cache(maxsize=1)
def get_azure_openai_settings() -> AzureOpenAISettings:
    """Load Azure OpenAI settings once from the local environment file."""

    values = _load_env_values()

    return AzureOpenAISettings(
        api_key=_env(values, "AZURE_OPENAI_API_KEY"),
        endpoint=_env(values, "AZURE_OPENAI_ENDPOINT"),
        deployment=_env(values, "AZURE_OPENAI_DEPLOYMENT"),
        api_version=_env(values, "AZURE_OPENAI_API_VERSION"),
        temperature=_float_env(values, "AZURE_OPENAI_TEMPERATURE", "0"),
        max_tokens=_int_env(values, "AZURE_OPENAI_MAX_TOKENS", "5000"),
    )


@lru_cache(maxsize=1)
def get_user_simulator_foundry_settings() -> UserSimulatorFoundrySettings:
    """Load Anthropic Foundry settings for the user simulator from environment values."""

    values = _load_env_values()

    return UserSimulatorFoundrySettings(
        endpoint=_env(values, "USER_SIMULATOR_FOUNDRY_ENDPOINT"),
        api_key=_env(values, "USER_SIMULATOR_FOUNDRY_API_KEY"),
        model=_env(values, "USER_SIMULATOR_FOUNDRY_MODEL"),
        temperature=_float_env(values, "USER_SIMULATOR_FOUNDRY_TEMPERATURE", "0"),
        max_tokens=_int_env(values, "USER_SIMULATOR_FOUNDRY_MAX_TOKENS", "5000"),
    )


@lru_cache(maxsize=1)
def get_llm_judge_azure_openai_settings() -> LlmJudgeSettings:
    """Load Azure OpenAI settings for the regression judge from environment values."""

    values = _load_env_values()

    # The judge currently falls back to the same model config as the agent.
    # A production setup should use separate judge-specific config values.
    return LlmJudgeSettings(
        api_key=(
            _env(values, "LLM_JUDGE_AZURE_OPENAI_API_KEY")
            or _env(values, "AZURE_OPENAI_API_KEY")
        ),
        endpoint=(
            _env(values, "LLM_JUDGE_AZURE_OPENAI_ENDPOINT")
            or _env(values, "AZURE_OPENAI_ENDPOINT")
        ),
        deployment=(
            _env(values, "LLM_JUDGE_AZURE_OPENAI_DEPLOYMENT")
            or _env(values, "AZURE_OPENAI_DEPLOYMENT")
        ),
        api_version=(
            _env(values, "LLM_JUDGE_AZURE_OPENAI_API_VERSION")
            or _env(values, "AZURE_OPENAI_API_VERSION")
        ),
        temperature=float(
            _env(values, "LLM_JUDGE_AZURE_OPENAI_TEMPERATURE")
            or _env(values, "AZURE_OPENAI_TEMPERATURE")
            or "0"
        ),
        max_tokens=int(
            _env(values, "LLM_JUDGE_AZURE_OPENAI_MAX_TOKENS")
            or _env(values, "AZURE_OPENAI_MAX_TOKENS")
            or "5000"
        ),
    )


@lru_cache(maxsize=1)
def get_mlflow_settings() -> MlflowSettings:
    """Load MLflow settings from environment values."""

    values = _load_env_values()
    raw_tracking_mode = _env(values, "MLFLOW_TRACKING_MODE")
    if not raw_tracking_mode:
        raise ValueError("MLFLOW_TRACKING_MODE is required: local, azureml, or remote")

    tracking_mode = MlflowTrackingMode(raw_tracking_mode.lower())
    experiment_name = _env(values, "MLFLOW_EXPERIMENT_NAME") or "company-fit-check"
    common_settings = {
        "experiment_name": experiment_name,
    }

    if tracking_mode is MlflowTrackingMode.LOCAL:
        tracking_uri = _env(values, "MLFLOW_TRACKING_URI")
        artifact_root = _env(values, "MLFLOW_ARTIFACT_ROOT")
        azure_storage_connection_string = _env(values, "AZURE_STORAGE_CONNECTION_STRING")

        if not tracking_uri:
            raise ValueError(
                "MLFLOW_TRACKING_URI is required when MLFLOW_TRACKING_MODE=local"
            )
        if not tracking_uri.startswith("sqlite:///"):
            raise ValueError(
                "MLFLOW_TRACKING_MODE=local requires a sqlite:/// tracking URI"
            )
        if not artifact_root:
            raise ValueError(
                "MLFLOW_ARTIFACT_ROOT is required when MLFLOW_TRACKING_MODE=local"
            )
        if artifact_root.startswith("wasbs://") and not azure_storage_connection_string:
            raise ValueError(
                "AZURE_STORAGE_CONNECTION_STRING is required when local MLflow "
                "artifacts use wasbs://"
            )

        return LocalMlflowSettings(
            tracking_uri=tracking_uri,
            artifact_root=artifact_root,
            azure_storage_connection_string=azure_storage_connection_string,
            **common_settings,
        )

    if tracking_mode is MlflowTrackingMode.AZUREML:
        tracking_uri = _env(values, "MLFLOW_TRACKING_URI")
        if not tracking_uri:
            raise ValueError(
                "MLFLOW_TRACKING_URI is required when MLFLOW_TRACKING_MODE=azureml"
            )
        if not tracking_uri.startswith("azureml://"):
            raise ValueError(
                "MLFLOW_TRACKING_MODE=azureml requires an azureml:// tracking URI"
            )

        return AzureMlflowSettings(
            tracking_uri=tracking_uri,
            **common_settings,
        )

    tracking_uri = _env(values, "MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError(
            "MLFLOW_TRACKING_URI is required when MLFLOW_TRACKING_MODE=remote"
        )
    if not tracking_uri.startswith(("http://", "https://")):
        raise ValueError(
            "MLFLOW_TRACKING_MODE=remote requires an http:// or https:// tracking URI"
        )

    return RemoteMlflowSettings(
        tracking_uri=tracking_uri,
        **common_settings,
    )


@lru_cache(maxsize=1)
def get_logging_settings() -> LoggingSettings:
    """Load logging settings from environment values."""

    values = _load_env_values()
    return LoggingSettings(
        level=_env(values, "COMPANY_FIT_CHECK_LOG_LEVEL") or "INFO",
    )
