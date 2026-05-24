"""JSON case loading for deterministic evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from evals import eval_root
from evals.models import EvalCase


def load_eval_cases(case_dir: Path | None = None) -> list[EvalCase]:
    """Load all deterministic evaluation cases from disk."""

    root = case_dir or (eval_root() / "deterministic_cases")
    cases: list[EvalCase] = []
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases.append(EvalCase.model_validate(payload))
    return cases
