"""Pytest entrypoint for live non-deterministic LLM regression runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import (
    get_azure_openai_settings,
    get_user_simulator_foundry_settings,
)
from evals.regression_runner import run_case


def _llm_regression_is_configured() -> bool:
    """Return whether both agent and user-simulator deployments are configured."""

    return (
        get_azure_openai_settings().is_configured
        and get_user_simulator_foundry_settings().is_configured
    )


@pytest.mark.skipif(
    not _llm_regression_is_configured(),
    reason="Live LLM regression requires both agent Azure OpenAI and user simulator Anthropic Foundry config.",
)
def test_llm_regression_case(regression_case_id: int):
    result = run_case(regression_case_id)

    assert result.run_id
    assert result.status == "completed", result.model_dump_json(indent=2)
    assert Path(result.transcript_path).exists()
