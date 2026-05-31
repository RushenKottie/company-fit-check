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
from evals.mutation_case_generator import MUTATION_CASE_COUNT


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add CLI options for non-deterministic regression selection."""

    parser.addoption(
        "--case-ids",
        action="store",
        default="",
        help="Comma-separated non-deterministic regression case ids to run.",
    )
    parser.addoption(
        "--concurrent",
        action="store_true",
        default=False,
        help="Run selected live regression or mutation cases concurrently.",
    )
    parser.addoption(
        "--max-workers",
        action="store",
        type=int,
        default=None,
        help="Maximum concurrent workers for live regression or mutation cases.",
    )
    parser.addoption(
        "--mutation-count",
        action="store",
        type=int,
        default=MUTATION_CASE_COUNT,
        help="Number of live mutation cases to generate and run.",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize regression case ids from the user simulator inventory."""

    if (
        "regression_case_id" not in metafunc.fixturenames
        and "regression_case_ids" not in metafunc.fixturenames
    ):
        return

    selected_ids = parse_case_ids_option(metafunc.config.getoption("case_ids"))
    available_ids = list_available_case_ids()
    case_ids = selected_ids or available_ids
    unknown_ids = sorted(set(case_ids) - set(available_ids))
    if unknown_ids:
        raise pytest.UsageError(
            f"Unknown regression case ids requested: {', '.join(str(case_id) for case_id in unknown_ids)}"
        )

    if "regression_case_ids" in metafunc.fixturenames:
        if metafunc.config.getoption("concurrent"):
            metafunc.parametrize("regression_case_ids", [case_ids], ids=["concurrent_cases"])
        else:
            metafunc.parametrize(
                "regression_case_ids",
                [[case_id] for case_id in case_ids],
                ids=[f"case_{case_id}" for case_id in case_ids],
            )
        return

    metafunc.parametrize(
        "regression_case_id",
        case_ids,
        ids=[f"case_{case_id}" for case_id in case_ids],
    )


@pytest.fixture
def eval_concurrent(pytestconfig: pytest.Config) -> bool:
    """Return whether live eval cases should be run concurrently."""

    return bool(pytestconfig.getoption("concurrent"))


@pytest.fixture
def eval_max_workers(pytestconfig: pytest.Config) -> int | None:
    """Return the configured live eval worker limit."""

    max_workers = pytestconfig.getoption("max_workers")
    if max_workers is not None and max_workers < 1:
        raise pytest.UsageError("--max-workers must be a positive integer.")
    return max_workers


@pytest.fixture
def eval_mutation_count(pytestconfig: pytest.Config) -> int:
    """Return the configured live mutation case count."""

    mutation_count = pytestconfig.getoption("mutation_count")
    if mutation_count < 1:
        raise pytest.UsageError("--mutation-count must be a positive integer.")
    return mutation_count


def parse_case_ids_option(raw_value: str) -> list[int]:
    """Parse one comma-separated case id option into a stable list of ints."""

    if not raw_value.strip():
        return []

    values = [part.strip() for part in raw_value.split(",") if part.strip()]
    try:
        return [int(value) for value in values]
    except ValueError as exc:
        raise pytest.UsageError("--case-ids must contain only integers separated by commas.") from exc
