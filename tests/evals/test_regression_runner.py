"""Unit tests for the non-deterministic regression runner."""

from __future__ import annotations

import json
from pathlib import Path
from contextlib import contextmanager

import conftest

from application.session import SessionResult
from evals.nondeterministic.case_models import NondeterministicCase
from evals.nondeterministic.models import NondeterministicRunResult
from evals.nondeterministic.runner import NondeterministicRunner
from evals.user_simulator.models import ReplyToAgentResponse, StartCaseResponse
from models.artifacts import GeneratedArtifact


class FakeSimulator:
    """Small simulator fake for runner tests."""

    def __init__(self, pdf_path: str = "fake.pdf") -> None:
        self._pdf_path = pdf_path

    def start_case(self, request):
        return StartCaseResponse(
            case_id=request.case_id,
            case_name="fake_case",
            prompt="initial prompt",
            pdf_path=self._pdf_path,
        )

    def reply_to_agent(self, request):
        return ReplyToAgentResponse(
            run_id=request.run_id,
            case_id=request.case_id,
            case_name="fake_case",
            answer="clarification answer",
        )


def _fake_case_index() -> dict[int, NondeterministicCase]:
    return {
        7: NondeterministicCase(
            id=7,
            name="fake_case",
            profession="Fake Engineer",
            experience="5 years",
            goal="Find matching companies.",
            pdf_path="fake.pdf",
            filter_criteria=["remote friendly"],
            axes=["engineering culture"],
            communication_style={
                "description": "direct",
                "behavioral_traits": ["concise"],
            },
            first_prompt="initial prompt",
        )
    }


def test_parse_case_ids_option_handles_empty_and_csv_values():
    assert conftest.parse_case_ids_option("") == []
    assert conftest.parse_case_ids_option("1, 8,10") == [1, 8, 10]


def test_runner_writes_transcript_and_csv_for_completed_case(tmp_path, monkeypatch):
    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    def fake_start_session(cv_pdf_bytes: bytes, prompt: str) -> SessionResult:
        assert cv_pdf_bytes == b"%PDF-1.4 fake"
        assert prompt == "initial prompt"
        return SessionResult(
            state={
                "run_id": "run-123",
                "session_status": "needs_clarification",
                "pending_clarification_message": "Need more detail?",
            }
        )

    def fake_continue_session(state, user_response: str) -> SessionResult:
        assert state["run_id"] == "run-123"
        assert user_response == "clarification answer"
        return SessionResult(
            state={
                "run_id": "run-123",
                "session_status": "completed",
                "companies": [],
                "company_scores": [],
                "axes": [],
            },
            csv_artifact=GeneratedArtifact(
                filename="results.csv",
                content_type="text/csv",
                content_bytes=b"company_name,overall_score\n",
            ),
        )

    monkeypatch.setattr("evals.nondeterministic.runner.start_session", fake_start_session)
    monkeypatch.setattr("evals.nondeterministic.runner.continue_session", fake_continue_session)

    runner = NondeterministicRunner(
        user_simulator=FakeSimulator(str(pdf_path)),
        case_index=_fake_case_index(),
        artifact_root=tmp_path,
    )
    result = runner.run_case(7, suite_stamp="suite")

    assert result == NondeterministicRunResult(
        case_id=7,
        case_name="fake_case",
        run_id="run-123",
        status="completed",
        turn_count=4,
        transcript_path=result.transcript_path,
        csv_artifact_path=result.csv_artifact_path,
        error=None,
    )
    transcript = json.loads(Path(result.transcript_path).read_text(encoding="utf-8"))
    assert transcript["run_id"] == "run-123"
    assert transcript["status"] == "completed"
    assert [turn["speaker"] for turn in transcript["turns"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert Path(result.csv_artifact_path).read_text(encoding="utf-8") == "company_name,overall_score\n"


def test_runner_binds_regression_mlflow_experiment(tmp_path, monkeypatch):
    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    bound_experiments: list[str] = []

    @contextmanager
    def fake_bind_mlflow_experiment(experiment_name: str):
        bound_experiments.append(experiment_name)
        yield

    def fake_start_session(cv_pdf_bytes: bytes, prompt: str) -> SessionResult:
        return SessionResult(
            state={
                "run_id": "run-123",
                "session_status": "completed",
                "companies": [],
                "company_scores": [],
                "axes": [],
            }
        )

    monkeypatch.setattr("evals.nondeterministic.runner.bind_mlflow_experiment", fake_bind_mlflow_experiment)
    monkeypatch.setattr("evals.nondeterministic.runner.start_session", fake_start_session)

    runner = NondeterministicRunner(
        user_simulator=FakeSimulator(str(pdf_path)),
        case_index=_fake_case_index(),
        artifact_root=tmp_path,
    )
    runner.run_case(7, suite_stamp="suite")

    assert bound_experiments == ["company-fit-check-llm-regression"]


def test_run_cases_preserves_requested_order_in_concurrent_mode(tmp_path):
    class RecordingRunner(NondeterministicRunner):
        def run_case(self, case_id: int, *, suite_stamp: str | None = None) -> NondeterministicRunResult:
            return NondeterministicRunResult(
                case_id=case_id,
                case_name=f"case_{case_id}",
                run_id=f"run-{case_id}",
                status="completed",
                turn_count=0,
                transcript_path=str(tmp_path / f"{case_id}.json"),
            )

    runner = RecordingRunner(
        user_simulator=FakeSimulator(),
        case_index=_fake_case_index(),
        artifact_root=tmp_path,
    )
    results = runner.run_cases([4, 2, 9], concurrent=True, max_workers=3)

    assert [result.case_id for result in results] == [4, 2, 9]
