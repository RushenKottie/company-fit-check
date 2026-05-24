"""Deterministic evaluation helpers and constants."""

from pathlib import Path

EVAL_EXPERIMENT_NAME = "company-fit-check-deterministic-evals"
REGRESSION_EXPERIMENT_NAME = "company-fit-check-llm-regression"


def repo_root() -> Path:
    """Return the repository root."""

    return Path(__file__).resolve().parents[2]


def eval_root() -> Path:
    """Return the repository-local eval fixture root."""

    return repo_root() / "eval_data"
