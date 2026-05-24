"""JSON loading for non-deterministic regression case files."""

from __future__ import annotations

import json
from pathlib import Path

from evals import eval_root
from evals.non_deterministic_models import NonDeterministicRegressionCase


def load_non_deterministic_regression_cases(
    case_dir: Path | None = None,
) -> list[NonDeterministicRegressionCase]:
    """Load all non-deterministic regression case files from disk."""

    root = case_dir or (eval_root() / "non_deterministic_regression")
    cases: list[NonDeterministicRegressionCase] = []
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases.append(NonDeterministicRegressionCase.model_validate(payload))
    return cases
