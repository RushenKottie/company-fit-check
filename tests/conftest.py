"""Shared pytest setup."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from user_simulator import list_available_case_ids


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add CLI options for non-deterministic regression selection."""

    parser.addoption(
        "--case-ids",
        action="store",
        default="",
        help="Comma-separated non-deterministic regression case ids to run.",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize regression case ids from the user simulator inventory."""

    if "regression_case_id" not in metafunc.fixturenames:
        return

    selected_ids = parse_case_ids_option(metafunc.config.getoption("case_ids"))
    available_ids = list_available_case_ids()
    case_ids = selected_ids or available_ids
    unknown_ids = sorted(set(case_ids) - set(available_ids))
    if unknown_ids:
        raise pytest.UsageError(
            f"Unknown regression case ids requested: {', '.join(str(case_id) for case_id in unknown_ids)}"
        )

    metafunc.parametrize("regression_case_id", case_ids, ids=[f"case_{case_id}" for case_id in case_ids])


def parse_case_ids_option(raw_value: str) -> list[int]:
    """Parse one comma-separated case id option into a stable list of ints."""

    if not raw_value.strip():
        return []

    values = [part.strip() for part in raw_value.split(",") if part.strip()]
    try:
        return [int(value) for value in values]
    except ValueError as exc:
        raise pytest.UsageError("--case-ids must contain only integers separated by commas.") from exc
