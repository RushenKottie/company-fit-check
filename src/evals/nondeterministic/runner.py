"""Non-deterministic runner using the shared session entrypoint."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import uuid
from typing import Sequence

from application.session import continue_session, start_session
from config import get_mlflow_settings
from evals import eval_root, repo_root
from evals.nondeterministic.case_loader import build_nondeterministic_case_index
from evals.nondeterministic.case_models import NondeterministicCase
from evals.nondeterministic.judge import (
    get_default_nondeterministic_judge_config,
    judge_nondeterministic_case,
)
from evals.nondeterministic.models import (
    NondeterministicJudgeConfig,
    NondeterministicJudgeOutcome,
    NondeterministicJudgeRequest,
    NondeterministicJudgeResponse,
    NondeterministicRunResult,
    TranscriptTurn,
)
from evals.nondeterministic.run_outputs import (
    rewrite_transcript_status,
    suite_stamp as build_suite_stamp,
    write_csv_artifact,
    write_transcript,
)
from infrastructure.mlflow_tracking import (
    bind_mlflow_experiment,
    create_run_id,
    ensure_case_dataset_for_run,
    log_json_artifact_for_run,
    log_metric_for_run,
    set_run_status_for_run,
    set_run_name_for_run,
)
from logging_utils import configure_logging, get_logger
from models.input import UserInput
from models.state import (
    SESSION_STATUS_COMPLETED,
    SESSION_STATUS_FAILED,
    SESSION_STATUS_NEEDS_CLARIFICATION,
)
from services.pdf_text import extract_text_from_pdf_bytes
from evals.user_simulator import (
    ConversationTurn,
    ReplyToAgentRequest,
    StartCaseRequest,
    UserSimulator,
)


logger = get_logger(__name__)
_RUN_FAILURE_THRESHOLD = 3


@dataclass(slots=True)
class NondeterministicRunner:
    """Run non-deterministic conversations against the agent."""

    user_simulator: UserSimulator
    case_index: Mapping[int, NondeterministicCase]
    artifact_root: Path
    experiment_name: str | None = None
    case_source: str = "regression"
    max_clarification_turns: int = 8

    def run_case(
        self,
        case_id: int,
        *,
        suite_stamp: str | None = None,
    ) -> NondeterministicRunResult:
        """Run one non-deterministic case end to end and write transcript artifacts."""

        configure_logging()
        resolved_suite_stamp = suite_stamp or build_suite_stamp()
        turns: list[TranscriptTurn] = []
        case_name = f"case_{case_id}"
        pdf_path = ""
        run_id = ""
        status = SESSION_STATUS_FAILED
        error: str | None = None
        csv_artifact_path: str | None = None

        try:
            with bind_mlflow_experiment(
                self.experiment_name or get_mlflow_settings().regression_experiment_name
            ):
                case = self._get_case(case_id)
                start_response = self.user_simulator.start_case(StartCaseRequest(case_id=case_id))
                case_name = start_response.case_name
                pdf_path = start_response.pdf_path
                turns.append(_turn("user", start_response.prompt))
                pdf_bytes = _resolve_pdf_path(pdf_path).read_bytes()
                run_id = create_run_id(
                    UserInput(cv_pdf_bytes=pdf_bytes, prompt=start_response.prompt)
                )
                set_run_name_for_run(
                    run_id,
                    _build_nondeterministic_run_name(
                        case_id=case_id,
                        profession=case.profession,
                        run_id=run_id,
                    ),
                )
                ensure_case_dataset_for_run(
                    run_id,
                    case_id=case.id,
                    case_name=case.name,
                    case_payload=case.model_dump(mode="json"),
                    case_source=self.case_source,
                )

                session_result = start_session(
                    pdf_bytes,
                    start_response.prompt,
                    run_id=run_id,
                )

                assistant_message = session_result.assistant_message
                turns.append(_turn("assistant", assistant_message))

                clarification_turns = 0
                while (
                    session_result.state.get("session_status")
                    == SESSION_STATUS_NEEDS_CLARIFICATION
                ):
                    if clarification_turns >= self.max_clarification_turns:
                        status = SESSION_STATUS_FAILED
                        error = "max_clarification_turns_exceeded"
                        break

                    simulator_reply = self.user_simulator.reply_to_agent(
                        ReplyToAgentRequest(
                            run_id=run_id,
                            case_id=case_id,
                            agent_message=assistant_message,
                            conversation=_simulator_conversation(turns[:-1]),
                        )
                    )
                    turns.append(_turn("user", simulator_reply.answer))

                    session_result = continue_session(session_result.state, simulator_reply.answer)
                    assistant_message = session_result.assistant_message
                    turns.append(_turn("assistant", assistant_message))
                    clarification_turns += 1

                if error is None:
                    status = session_result.state.get(
                        "session_status",
                        SESSION_STATUS_FAILED,
                    )
                    error = session_result.state.get("error")
                    if (
                        session_result.csv_artifact is not None
                        and status == SESSION_STATUS_COMPLETED
                    ):
                        csv_artifact_path = write_csv_artifact(
                            artifact_root=self.artifact_root,
                            suite_stamp=resolved_suite_stamp,
                            case_id=case_id,
                            run_id=run_id,
                            artifact=session_result.csv_artifact,
                        )
        except Exception as exc:
            error = _format_exception(exc)
            logger.exception("Regression runner failed case_id=%s", case_id)
            if not run_id:
                run_id = uuid.uuid4().hex

        transcript_path = write_transcript(
            artifact_root=self.artifact_root,
            suite_stamp=resolved_suite_stamp,
            case_id=case_id,
            case_name=case_name,
            run_id=run_id,
            pdf_path=pdf_path,
            turns=turns,
            status=status,
            error=error,
        )
        return NondeterministicRunResult(
            case_id=case_id,
            case_name=case_name,
            run_id=run_id,
            status=status,
            turn_count=len(turns),
            transcript_path=transcript_path,
            csv_artifact_path=csv_artifact_path,
            error=error,
        )

    def run_cases(
        self,
        case_ids: Sequence[int],
        *,
        concurrent: bool = False,
        max_workers: int | None = None,
    ) -> list[NondeterministicRunResult]:
        """Run multiple cases sequentially or in parallel, preserving request order."""

        suite_stamp = build_suite_stamp()
        ordered_case_ids = list(case_ids)
        if not concurrent or len(ordered_case_ids) < 2:
            results = [
                self.run_case(case_id, suite_stamp=suite_stamp)
                for case_id in ordered_case_ids
            ]
            return self._judge_results(results)

        worker_count = max_workers or min(len(ordered_case_ids), 4)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(
                executor.map(
                    lambda case_id: self.run_case(case_id, suite_stamp=suite_stamp),
                    ordered_case_ids,
                )
            )
        return self._judge_results(results)

    def _get_case(self, case_id: int) -> NondeterministicCase:
        """Return one runner case from the injected source."""

        case = self.case_index.get(case_id)
        if case is None:
            raise ValueError(f"Unknown non-deterministic case id: {case_id}.")
        return case

    def _judge_results(
        self,
        results: list[NondeterministicRunResult],
    ) -> list[NondeterministicRunResult]:
        """Run the second-phase LLM judge over all case results."""

        judge_config = get_default_nondeterministic_judge_config()
        for result in results:
            self._judge_one_result(result, judge_config=judge_config)
        return results

    def _judge_one_result(
        self,
        result: NondeterministicRunResult,
        *,
        judge_config: NondeterministicJudgeConfig,
    ) -> None:
        """Judge one non-deterministic result and log metrics to the same MLflow run."""

        request: NondeterministicJudgeRequest | None = None
        llm_called = False
        try:
            transcript_path = Path(result.transcript_path)
            transcript_payload = json.loads(transcript_path.read_text(encoding="utf-8"))

            csv_path = _existing_csv_path(result.csv_artifact_path)
            if csv_path is None:
                outcome = _missing_csv_outcome()
                self._log_judge_outcome(result.run_id, outcome)
                self._mark_unjudged_result_failed(result, outcome)
                return

            request = _build_judge_request(
                result,
                transcript_payload,
                csv_path,
                judge_config,
            )
            llm_called = True
            response = judge_nondeterministic_case(request)
        except Exception as exc:
            outcome = NondeterministicJudgeOutcome(
                judge_status=SESSION_STATUS_FAILED,
                judge_error=_format_exception(exc),
                llm_called=llm_called,
                request=request.model_dump(mode="json") if request is not None else None,
            )
            self._log_judge_outcome(result.run_id, outcome)
            _apply_judge_outcome_to_result(result, outcome)
            if not llm_called:
                self._mark_unjudged_result_failed(result, outcome)
            return

        judge_error = _build_judge_threshold_failure_error(response)
        outcome = NondeterministicJudgeOutcome(
            judge_status=(
                SESSION_STATUS_FAILED
                if judge_error is not None
                else SESSION_STATUS_COMPLETED
            ),
            judge_error=judge_error,
            llm_called=True,
            request=request.model_dump(mode="json"),
            response=response.model_dump(mode="json"),
        )
        self._log_judge_outcome(result.run_id, outcome)
        _log_judge_metrics(result.run_id, response)
        overall_score = _compute_overall_score(response)
        if overall_score is not None:
            log_metric_for_run(
                result.run_id,
                "overall_score",
                overall_score,
            )
        if judge_error is not None:
            result.status = SESSION_STATUS_FAILED
            result.error = judge_error
            rewrite_transcript_status(
                result.transcript_path,
                status=result.status,
                error=result.error,
            )
            set_run_status_for_run(
                result.run_id,
                status=result.status,
                error=result.error,
            )
        _apply_judge_outcome_to_result(result, outcome)

    def _mark_unjudged_result_failed(
        self,
        result: NondeterministicRunResult,
        outcome: NondeterministicJudgeOutcome,
    ) -> None:
        """Mark a run failed when execution prevented LLM judging."""

        result.status = SESSION_STATUS_FAILED
        result.error = result.error or outcome.judge_error
        _apply_judge_outcome_to_result(result, outcome)
        rewrite_transcript_status(
            result.transcript_path,
            status=result.status,
            error=result.error,
        )
        set_run_status_for_run(
            result.run_id,
            status=result.status,
            error=result.error,
        )

    def _log_judge_outcome(self, run_id: str, outcome: NondeterministicJudgeOutcome) -> None:
        """Persist judge request/result metadata onto the existing run."""

        if outcome.request is not None:
            log_json_artifact_for_run(run_id, "judge/request.json", outcome.request)
        if outcome.response is not None:
            log_json_artifact_for_run(run_id, "judge/result.json", outcome.response)
        log_json_artifact_for_run(
            run_id,
            "judge/outcome.json",
            outcome.model_dump(mode="json"),
        )

def create_default_nondeterministic_runner() -> NondeterministicRunner:
    """Return the default non-deterministic runner instance."""

    settings = get_mlflow_settings()
    case_index = build_nondeterministic_case_index(
        eval_root() / "non_deterministic_regression"
    )
    return NondeterministicRunner(
        user_simulator=UserSimulator(case_index=case_index),
        case_index=case_index,
        artifact_root=(repo_root() / "artifacts" / "regression").resolve(),
        experiment_name=settings.regression_experiment_name,
        case_source="regression",
    )


def create_mutation_runner(case_dir: Path) -> NondeterministicRunner:
    """Return a non-deterministic runner backed by generated mutation cases."""

    settings = get_mlflow_settings()
    case_index = build_nondeterministic_case_index(case_dir)
    return NondeterministicRunner(
        user_simulator=UserSimulator(case_index=case_index),
        case_index=case_index,
        artifact_root=(repo_root() / "artifacts" / "mutation_tests" / "runs").resolve(),
        experiment_name=settings.mutation_experiment_name,
        case_source="mutation",
    )


def _resolve_pdf_path(pdf_path: str) -> Path:
    """Resolve one runner-facing PDF path against the repository root."""

    path = Path(pdf_path)
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def _existing_csv_path(csv_artifact_path: str | None) -> Path | None:
    """Return the CSV path when it is present on disk."""

    if not csv_artifact_path:
        return None
    path = Path(csv_artifact_path)
    return path if path.exists() else None


def _missing_csv_outcome() -> NondeterministicJudgeOutcome:
    """Return the judge outcome used when a run has no CSV to judge."""

    return NondeterministicJudgeOutcome(
        judge_status=SESSION_STATUS_FAILED,
        judge_error="generated_csv_missing",
        llm_called=False,
    )


def _apply_judge_outcome_to_result(
    result: NondeterministicRunResult,
    outcome: NondeterministicJudgeOutcome,
) -> None:
    """Copy judge outcome fields onto a run result."""

    result.judge_status = outcome.judge_status
    result.judge_error = outcome.judge_error
    result.judge_result = outcome.model_dump(mode="json")


def _build_judge_request(
    result: NondeterministicRunResult,
    transcript_payload: dict,
    csv_path: Path,
    judge_config: NondeterministicJudgeConfig,
) -> NondeterministicJudgeRequest:
    """Build the request payload sent to the LLM judge."""

    pdf_path = transcript_payload.get("pdf_path", "")
    raw_cv_text = extract_text_from_pdf_bytes(_resolve_pdf_path(pdf_path).read_bytes())
    return NondeterministicJudgeRequest(
        case_id=result.case_id,
        case_name=result.case_name,
        run_id=result.run_id,
        status=result.status,
        initial_prompt=_extract_initial_prompt(transcript_payload),
        unmasked_cv_text=raw_cv_text,
        transcript_json=json.dumps(transcript_payload, ensure_ascii=True, indent=2),
        generated_csv=csv_path.read_text(encoding="utf-8"),
        judge_system_prompt=judge_config.system_prompt,
        judge_metrics=judge_config.metrics,
    )


def _log_judge_metrics(
    run_id: str,
    response: NondeterministicJudgeResponse,
) -> None:
    """Log the component scores returned by the judge."""

    log_metric_for_run(
        run_id,
        "clarification_quality",
        response.clarification_quality,
    )
    log_metric_for_run(
        run_id,
        "assumption_control",
        response.assumption_control,
    )
    log_metric_for_run(
        run_id,
        "reasoning_relevance_constraint_alignment",
        response.reasoning_relevance_constraint_alignment,
    )


def _compute_overall_score(response: NondeterministicJudgeResponse) -> float | None:
    """Return the average judge score, or None when any component is below threshold."""

    component_scores = _judge_component_scores(response)
    if any(score < _RUN_FAILURE_THRESHOLD for _, score in component_scores):
        return None
    return round(
        sum(score for _, score in component_scores) / len(component_scores),
        1,
    )


def _build_judge_threshold_failure_error(
    response: NondeterministicJudgeResponse,
) -> str | None:
    """Return one error message when any judge metric falls below threshold."""

    failed_metrics = [
        f"{name}={score}"
        for name, score in _judge_component_scores(response)
        if score < _RUN_FAILURE_THRESHOLD
    ]
    if not failed_metrics:
        return None
    return (
        "judge_component_below_threshold: "
        f"{', '.join(failed_metrics)}; "
        f"threshold={_RUN_FAILURE_THRESHOLD}"
    )


def _judge_component_scores(
    response: NondeterministicJudgeResponse,
) -> list[tuple[str, int]]:
    """Return the named metric scores from one judge response."""

    return [
        ("clarification_quality", response.clarification_quality),
        ("assumption_control", response.assumption_control),
        (
            "reasoning_relevance_constraint_alignment",
            response.reasoning_relevance_constraint_alignment,
        ),
    ]


def _build_nondeterministic_run_name(*, case_id: int, profession: str, run_id: str) -> str:
    """Return a compact MLflow display name for one non-deterministic case run."""

    prefix = re.sub(r"[^a-z0-9]+", "_", profession.strip().lower()).strip("_") or "case"
    return f"{prefix}_{case_id}_{run_id[:8]}"


def _turn(speaker: str, message: str) -> TranscriptTurn:
    """Build one transcript turn with a UTC timestamp."""

    return TranscriptTurn(
        speaker=speaker,
        message=message,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


def _simulator_conversation(turns: list[TranscriptTurn]) -> list[ConversationTurn]:
    """Return prior turns in the compact shape expected by the user simulator."""

    return [
        ConversationTurn(speaker=turn.speaker, message=turn.message)
        for turn in turns
    ]


def _extract_initial_prompt(transcript_payload: dict) -> str:
    """Return the first user message from one transcript payload."""

    for turn in transcript_payload.get("turns", []):
        if turn.get("speaker") == "user":
            return str(turn.get("message", "")).strip()
    return ""


def _format_exception(exc: Exception) -> str:
    """Return a compact error message for runner failures."""

    message = str(exc).strip()
    if not message:
        return f"{exc.__class__.__name__} was raised without an error message."
    return f"{exc.__class__.__name__}: {message}"
