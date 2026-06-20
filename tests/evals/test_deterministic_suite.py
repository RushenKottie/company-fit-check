"""End-to-end deterministic evaluation suite."""

from __future__ import annotations

from dataclasses import dataclass, field
import json

import mlflow

from evals import EVAL_EXPERIMENT_NAME
from evals.deterministic.case_loader import load_eval_cases
from evals.deterministic.checks import run_requested_checks
from evals.deterministic.engine import execute_case
from evals.deterministic.mlflow_eval import (
    ensure_eval_experiment,
    log_json_artifact_to_active_run,
)
from evals.deterministic.models import CaseExecutionResult, CheckResult, EvalCase


@dataclass
class _SuiteState:
    """Mutable state accumulated while the deterministic suite runs."""

    failures: list[str] = field(default_factory=list)
    failure_details: list[dict[str, object]] = field(default_factory=list)
    summary_payload: dict[str, list[dict[str, object]]] = field(
        default_factory=lambda: {"cases": []},
    )
    suite_check_results: dict[str, float] = field(default_factory=dict)


def test_deterministic_suite():
    """Run all deterministic eval cases and fail if required checks fail."""

    experiment_id = ensure_eval_experiment()
    mlflow.set_experiment(EVAL_EXPERIMENT_NAME)
    suite_state = _SuiteState()

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="deterministic-suite",
        tags={
            "suite": "deterministic",
        },
    ) as suite_run:
        for case in load_eval_cases():
            _run_case(case, suite_run.info.run_id, suite_state)

        _log_suite_results(suite_state)
        _fail_suite_if_needed(suite_state)


def _run_case(case: EvalCase, run_id: str, suite_state: _SuiteState) -> None:
    """Execute one eval case and record its artifacts, metrics, and failures."""

    result = execute_case(case, run_id=run_id)
    check_results = run_requested_checks(
        result,
        [check.name for check in case.checks],
    )
    required_failures = _required_failures(case, check_results)

    _log_case_artifacts(case, result, check_results)
    _log_check_metrics(check_results, suite_state)
    _record_case_summary(case, result, required_failures, suite_state)
    _record_required_failures(case, required_failures, suite_state)
    _record_unexpected_exception(case, result, suite_state)


def _required_failures(
    case: EvalCase,
    check_results: list[CheckResult],
) -> list[CheckResult]:
    """Return the failed checks that should fail the suite."""

    return [
        check_result
        for check_spec, check_result in zip(case.checks, check_results, strict=False)
        if check_spec.required and not check_result.passed
    ]


def _log_case_artifacts(
    case: EvalCase,
    result: CaseExecutionResult,
    check_results: list[CheckResult],
) -> None:
    """Log the case execution result and check results as MLflow artifacts."""

    log_json_artifact_to_active_run(
        result.model_dump(mode="json"),
        f"cases/{case.id}/case-result.json",
    )
    log_json_artifact_to_active_run(
        {
            "checks": [check.model_dump(mode="json") for check in check_results],
        },
        f"cases/{case.id}/check-results.json",
    )


def _log_check_metrics(
    check_results: list[CheckResult],
    suite_state: _SuiteState,
) -> None:
    """Log check pass/fail metrics and store their suite-level values."""

    for check_result in check_results:
        metric_value = 1.0 if check_result.passed else 0.0
        suite_state.suite_check_results[check_result.name] = metric_value
        mlflow.log_metric(check_result.name, metric_value)


def _record_case_summary(
    case: EvalCase,
    result: CaseExecutionResult,
    required_failures: list[CheckResult],
    suite_state: _SuiteState,
) -> None:
    """Append one case summary row to the suite summary payload."""

    suite_state.summary_payload["cases"].append(
        {
            "case_id": case.id,
            "passed": not required_failures and result.uncaught_exception is None,
            "status": result.status,
            "error": result.error,
        }
    )


def _record_required_failures(
    case: EvalCase,
    required_failures: list[CheckResult],
    suite_state: _SuiteState,
) -> None:
    """Append required check failures to the suite failure collections."""

    for failed in required_failures:
        suite_state.failures.append(f"{case.id}: {failed.reason}")
        suite_state.failure_details.append(
            {
                "case_id": case.id,
                "check_name": failed.name,
                "message": failed.reason,
                "reason": failed.reason,
                "details": failed.details,
            }
        )


def _record_unexpected_exception(
    case: EvalCase,
    result: CaseExecutionResult,
    suite_state: _SuiteState,
) -> None:
    """Append an unexpected case exception to the suite failure collections."""

    if result.uncaught_exception is None:
        return

    suite_state.failures.append(
        f"{case.id}: uncaught exception {result.uncaught_exception}"
    )
    suite_state.failure_details.append(
        {
            "case_id": case.id,
            "check_name": "uncaught_exception",
            "message": result.uncaught_exception,
            "reason": result.uncaught_exception,
        }
    )


def _log_suite_results(suite_state: _SuiteState) -> None:
    """Log suite summary artifacts and aggregate MLflow metrics."""

    log_json_artifact_to_active_run(
        suite_state.summary_payload,
        "eval/suite-summary.json",
    )
    log_json_artifact_to_active_run(
        {
            "failures": suite_state.failure_details,
        },
        "eval/failures.json",
    )
    mlflow.log_metric(
        "suite.case_count",
        float(len(suite_state.summary_payload["cases"])),
    )
    mlflow.log_metric("suite.failure_count", float(len(suite_state.failures)))
    mlflow.log_metric("suite.passed", 0.0 if suite_state.failures else 1.0)
    log_json_artifact_to_active_run(
        {
            "checks": suite_state.suite_check_results,
        },
        "eval/suite-check-results.json",
    )


def _fail_suite_if_needed(suite_state: _SuiteState) -> None:
    """Mark the MLflow run failed and assert when the suite has failures."""

    if suite_state.failures:
        mlflow.end_run(status="FAILED")
        assert not suite_state.failures, json.dumps(suite_state.failures, indent=2)
