"""Runtime configuration helpers."""

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

from dotenv import dotenv_values, find_dotenv


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
    temperature: float | None
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


@dataclass(frozen=True)
class LoggingSettings:
    """Resolved logging settings for the app."""

    level: str


def _project_root() -> Path:
    """Return the repository root based on this module location."""

    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _load_env_values() -> dict[str, str]:
    """Resolve all environment-backed configuration values in one place."""

    env_path = find_dotenv(usecwd=True)
    file_values = dotenv_values(env_path) if env_path else {}
    merged = {
        key: normalized
        for key, value in file_values.items()
        if (normalized := _normalize_env_value(value)) is not None
    }
    for key, value in os.environ.items():
        if (normalized := _normalize_env_value(value)) is not None:
            merged[key] = normalized
    return merged


def _normalize_env_value(value: str | None) -> str | None:
    """Return a stripped env value, treating empty strings as missing."""

    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


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
def get_user_simulator_azure_openai_settings() -> AzureOpenAISettings:
    """Load Azure OpenAI settings for the user simulator from environment values."""

    values = _load_env_values()

    return AzureOpenAISettings(
        api_key=(
            _env(values, "USER_SIMULATOR_AZURE_OPENAI_API_KEY")
            or _env(values, "AZURE_OPENAI_API_KEY")
        ),
        endpoint=(
            _env(values, "USER_SIMULATOR_AZURE_OPENAI_ENDPOINT")
            or _env(values, "AZURE_OPENAI_ENDPOINT")
        ),
        deployment=_env(values, "USER_SIMULATOR_AZURE_OPENAI_DEPLOYMENT"),
        api_version=(
            _env(values, "USER_SIMULATOR_AZURE_OPENAI_API_VERSION")
            or _env(values, "AZURE_OPENAI_API_VERSION")
        ),
        temperature=float(
            _env(values, "USER_SIMULATOR_AZURE_OPENAI_TEMPERATURE")
            or _env(values, "AZURE_OPENAI_TEMPERATURE")
            or "0"
        ),
        max_tokens=int(
            _env(values, "USER_SIMULATOR_AZURE_OPENAI_MAX_TOKENS")
            or _env(values, "AZURE_OPENAI_MAX_TOKENS")
            or "5000"
        ),
    )


@lru_cache(maxsize=1)
def get_user_simulator_foundry_settings() -> UserSimulatorFoundrySettings:
    """Load Anthropic Foundry settings for the user simulator from environment values."""

    values = _load_env_values()
    temperature_value = _env(values, "USER_SIMULATOR_FOUNDRY_TEMPERATURE")

    return UserSimulatorFoundrySettings(
        endpoint=_env(values, "USER_SIMULATOR_FOUNDRY_ENDPOINT"),
        api_key=_env(values, "USER_SIMULATOR_FOUNDRY_API_KEY"),
        model=_env(values, "USER_SIMULATOR_FOUNDRY_MODEL"),
        temperature=float(temperature_value) if temperature_value else None,
        max_tokens=_int_env(values, "USER_SIMULATOR_FOUNDRY_MAX_TOKENS", "5000"),
    )


@lru_cache(maxsize=1)
def get_llm_judge_azure_openai_settings() -> LlmJudgeSettings:
    """Load Azure OpenAI settings for the regression judge from environment values."""

    values = _load_env_values()

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
    """Load MLflow and Azure Blob settings from environment values."""

    values = _load_env_values()
    default_tracking_path = (_project_root() / ".mlruns").resolve()
    tracking_uri = _env(values, "MLFLOW_TRACKING_URI") or default_tracking_path.as_uri()
    experiment_name = _env(values, "MLFLOW_EXPERIMENT_NAME") or "company-fit-check"
    regression_experiment_name = (
        _env(values, "MLFLOW_REGRESSION_EXPERIMENT_NAME")
        or "company-fit-check-llm-regression"
    )
    mutation_experiment_name = (
        _env(values, "MLFLOW_MUTATION_EXPERIMENT_NAME")
        or "company-fit-check-llm-mutation"
    )
    artifact_root = _env(values, "MLFLOW_ARTIFACT_ROOT")
    azure_storage_connection_string = (
        _env(values, "AZURE_STORAGE_CONNECTION_STRING")
    )

    return MlflowSettings(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        regression_experiment_name=regression_experiment_name,
        mutation_experiment_name=mutation_experiment_name,
        artifact_root=artifact_root,
        azure_storage_connection_string=azure_storage_connection_string,
    )


@lru_cache(maxsize=1)
def get_logging_settings() -> LoggingSettings:
    """Load logging settings from environment values."""

    values = _load_env_values()
    return LoggingSettings(
        level=_env(values, "COMPANY_FIT_CHECK_LOG_LEVEL") or "INFO",
    )
