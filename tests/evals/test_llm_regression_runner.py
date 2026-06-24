"""Pytest entrypoint for live non-deterministic LLM regression runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import (
    get_azure_openai_settings,
    get_user_simulator_foundry_settings,
)
from evals.nondeterministic.runner import create_default_nondeterministic_runner


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
def test_llm_regression_cases(
    regression_case_ids: list[int],
    eval_concurrent: bool,
    eval_max_workers: int | None,
):
    """Run live non-deterministic regression cases when LLM config is available."""

    results = create_default_nondeterministic_runner().run_cases(
        regression_case_ids,
        concurrent=eval_concurrent,
        max_workers=eval_max_workers,
    )

    for result in results:
        assert result.run_id
        assert result.status == "completed", result.model_dump_json(indent=2)
        assert Path(result.transcript_path).exists()
