"""JSON loading for non-deterministic evaluation case files."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

from evals import eval_root
from evals.nondeterministic.case_models import NondeterministicCase


def load_nondeterministic_cases(
    case_dir: Path | None = None,
) -> list[NondeterministicCase]:
    """Load all non-deterministic case files from disk."""

    root = case_dir or (eval_root() / "non_deterministic_regression")
    cases: list[NondeterministicCase] = []
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases.append(NondeterministicCase.model_validate(payload))
    return cases


def build_nondeterministic_case_index(
    case_dir: Path | None = None,
) -> MappingProxyType[int, NondeterministicCase]:
    """Load cases from disk and return a read-only ``case_id -> case`` map."""

    cases = load_nondeterministic_cases(case_dir)
    return MappingProxyType({case.id: case for case in cases})
