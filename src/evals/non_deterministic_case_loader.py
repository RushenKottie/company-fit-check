"""JSON loading for non-deterministic regression case files."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

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


def build_non_deterministic_case_index(
    case_dir: Path | None = None,
) -> MappingProxyType[int, NonDeterministicRegressionCase]:
    """Load cases from disk and return a read-only ``case_id -> case`` map."""

    cases = load_non_deterministic_regression_cases(case_dir)
    return MappingProxyType({case.id: case for case in cases})
