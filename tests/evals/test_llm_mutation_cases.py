"""Pytest entrypoint for live LLM mutation cases."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from config import (
    get_azure_openai_settings,
    get_user_simulator_foundry_settings,
)
from evals.mutation.case_generator import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PDF_OUTPUT_DIR,
    generate_mutation_case_files,
)
from evals.nondeterministic.case_loader import load_nondeterministic_cases
from evals.nondeterministic.runner import create_mutation_runner


def _llm_mutation_is_configured() -> bool:
    """Return whether both agent and user-simulator deployments are configured."""

    return (
        get_azure_openai_settings().is_configured
        and get_user_simulator_foundry_settings().is_configured
    )


@pytest.mark.skipif(
    not _llm_mutation_is_configured(),
    reason="Live mutation regression requires both agent Azure OpenAI and user simulator Anthropic Foundry config.",
)
def test_llm_mutation_cases(
    eval_concurrent: bool,
    eval_max_workers: int | None,
    eval_mutation_count: int,
):
    suite_stamp = _suite_stamp()
    case_dir = DEFAULT_OUTPUT_DIR / suite_stamp
    pdf_dir = DEFAULT_PDF_OUTPUT_DIR / suite_stamp

    generated_paths = generate_mutation_case_files(
        count=eval_mutation_count,
        output_dir=case_dir,
        pdf_output_dir=pdf_dir,
    )
    assert len(generated_paths) == eval_mutation_count

    cases = load_nondeterministic_cases(case_dir)
    assert len(cases) == eval_mutation_count

    case_ids = [case.id for case in cases]
    assert len(case_ids) == eval_mutation_count

    runner = create_mutation_runner(case_dir)
    results = runner.run_cases(
        case_ids,
        concurrent=eval_concurrent,
        max_workers=eval_max_workers,
    )
    assert len(results) == eval_mutation_count

    for result in results:
        assert result.run_id
        assert result.status == "completed", result.model_dump_json(indent=2)
        assert Path(result.transcript_path).exists()


def _suite_stamp() -> str:
    """Return one UTC timestamp slug for generated mutation cases."""

    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
