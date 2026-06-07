"""Data models for non-deterministic evaluation runs and judging."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TranscriptTurn(BaseModel):
    """One conversation turn recorded by the non-deterministic runner."""

    model_config = ConfigDict(extra="forbid")

    speaker: Literal["user", "assistant"]
    message: str
    timestamp_utc: str


class NondeterministicRunResult(BaseModel):
    """Normalized result for one non-deterministic conversation."""

    model_config = ConfigDict(extra="forbid")

    case_id: int
    case_name: str
    run_id: str
    status: str
    turn_count: int
    transcript_path: str
    csv_artifact_path: str | None = None
    error: str | None = None
    judge_status: str | None = None
    judge_error: str | None = None
    judge_result: dict[str, Any] | None = None


class JudgeMetricDefinition(BaseModel):
    """One metric definition sent to the LLM judge."""

    model_config = ConfigDict(extra="forbid")

    name: str
    rubric: str


class NondeterministicJudgeConfig(BaseModel):
    """Static prompt and metric configuration for the non-deterministic judge."""

    model_config = ConfigDict(extra="forbid")

    system_prompt: str
    metrics: list[JudgeMetricDefinition]


class NondeterministicJudgeRequest(BaseModel):
    """Case material sent to the LLM judge."""

    model_config = ConfigDict(extra="forbid")

    case_id: int
    case_name: str
    run_id: str
    status: str
    initial_prompt: str
    unmasked_cv_text: str
    transcript_json: str
    generated_csv: str
    judge_system_prompt: str
    judge_metrics: list[JudgeMetricDefinition]


class NondeterministicJudgeResponse(BaseModel):
    """Structured scoring response returned by the LLM judge."""

    model_config = ConfigDict(extra="forbid")

    clarification_quality: int = Field(ge=1, le=5)
    assumption_control: int = Field(ge=1, le=5)
    reasoning_relevance_constraint_alignment: int = Field(ge=1, le=5)
    clarification_quality_reason: str | None = None
    assumption_control_reason: str | None = None
    reasoning_relevance_constraint_alignment_reason: str | None = None


class NondeterministicJudgeOutcome(BaseModel):
    """Final judge outcome stored on the non-deterministic result."""

    model_config = ConfigDict(extra="forbid")

    judge_status: str
    judge_error: str | None = None
    llm_called: bool
    request: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
