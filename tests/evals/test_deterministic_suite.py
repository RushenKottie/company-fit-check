"""End-to-end deterministic evaluation suite."""

from __future__ import annotations

import json

import mlflow

from evals import EVAL_EXPERIMENT_NAME
from evals.case_loader import load_eval_cases
from evals.checks import run_requested_checks
from evals.engine import execute_case
from evals.mlflow_eval import (
    ensure_eval_experiment,
    log_json_artifact_to_active_run,
)


def test_deterministic_suite():
    experiment_id = ensure_eval_experiment()
    mlflow.set_experiment(EVAL_EXPERIMENT_NAME)
    failures: list[str] = []
    failure_details: list[dict[str, object]] = []
    summary_payload: dict[str, list[dict[str, object]]] = {"cases": []}
    suite_check_results: dict[str, float] = {}

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="deterministic-suite",
        tags={
            "suite": "deterministic",
        },
    ) as suite_run:
        for case in load_eval_cases():
            result = execute_case(case, run_id=suite_run.info.run_id)
            check_results = run_requested_checks(
                result,
                [check.name for check in case.checks],
            )
            required_failures = [
                check_result
                for check_spec, check_result in zip(case.checks, check_results, strict=False)
                if check_spec.required and not check_result.passed
            ]

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
            for check_result in check_results:
                suite_check_results[check_result.name] = (
                    1.0 if check_result.passed else 0.0
                )
                mlflow.log_metric(
                    check_result.name,
                    1.0 if check_result.passed else 0.0,
                )
            mlflow.log_metric(
                f"cases.{case.id}.passed",
                0.0 if required_failures or result.uncaught_exception else 1.0,
            )
            summary_payload["cases"].append(
                {
                    "case_id": case.id,
                    "passed": not required_failures and result.uncaught_exception is None,
                    "status": result.status,
                    "error": result.error,
                }
            )
            for failed in required_failures:
                failures.append(f"{case.id}: {failed.reason}")
                failure_details.append(
                    {
                        "case_id": case.id,
                        "check_name": failed.name,
                        "message": failed.reason,
                        "reason": failed.reason,
                        "details": failed.details,
                    }
                )
            if result.uncaught_exception is not None:
                failures.append(
                    f"{case.id}: uncaught exception {result.uncaught_exception}"
                )
                failure_details.append(
                    {
                        "case_id": case.id,
                        "check_name": "uncaught_exception",
                        "message": result.uncaught_exception,
                        "reason": result.uncaught_exception,
                    }
                )

        log_json_artifact_to_active_run(summary_payload, "eval/suite-summary.json")
        log_json_artifact_to_active_run(
            {
                "failures": failure_details,
            },
            "eval/failures.json",
        )
        mlflow.log_metric("suite.case_count", float(len(summary_payload["cases"])))
        mlflow.log_metric("suite.failure_count", float(len(failures)))
        mlflow.log_metric("suite.passed", 0.0 if failures else 1.0)
        log_json_artifact_to_active_run(
            {
                "checks": suite_check_results,
            },
            "eval/suite-check-results.json",
        )

    assert not failures, json.dumps(failures, indent=2)
