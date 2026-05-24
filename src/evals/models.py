"""Data models for deterministic evaluation cases and results."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EvalStubSpec(BaseModel):
    """One patch applied during case execution."""

    model_config = ConfigDict(extra="forbid")

    target: str
    stub: str


class EvalCheckSpec(BaseModel):
    """One requested deterministic check."""

    model_config = ConfigDict(extra="forbid")

    name: str
    required: bool = True


class EvalEntrypoint(BaseModel):
    """Case entrypoint definition."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["workflow", "node", "helper"]
    target: str


class EvalSetup(BaseModel):
    """Case setup definition."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["workflow_input", "state_snapshot"]
    pdf_path: str | None = None
    prompt: str | None = None
    state_path: str | None = None
    stubs: list[EvalStubSpec] = Field(default_factory=list)


class EvalCase(BaseModel):
    """One deterministic evaluation case."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    entrypoint: EvalEntrypoint
    setup: EvalSetup
    checks: list[EvalCheckSpec]
    clarifications: list[str] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)


class CheckResult(BaseModel):
    """Outcome of one deterministic check."""

    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)


class CaseExecutionResult(BaseModel):
    """Normalized execution result for one case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    description: str
    entrypoint_kind: str
    entrypoint_target: str
    run_id: str
    case_inputs: dict[str, Any] = Field(default_factory=dict)
    final_state_summary: dict[str, Any] = Field(default_factory=dict)
    helper_output: Any = None
    csv_artifact: dict[str, Any] | None = None
    generated_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    observed_spans: list[dict[str, Any]] = Field(default_factory=list)
    executed_span_names: list[str] = Field(default_factory=list)
    status: str
    error: str | None = None
    uncaught_exception: str | None = None


class TranscriptTurn(BaseModel):
    """One conversation turn recorded by the regression runner."""

    model_config = ConfigDict(extra="forbid")

    speaker: Literal["user", "assistant"]
    message: str
    timestamp_utc: str


class RegressionRunResult(BaseModel):
    """Normalized result for one non-deterministic regression conversation."""

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


class RegressionJudgeConfig(BaseModel):
    """Static prompt and metric configuration for the regression judge."""

    model_config = ConfigDict(extra="forbid")

    system_prompt: str
    metrics: list[JudgeMetricDefinition]


class RegressionJudgeRequest(BaseModel):
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


class RegressionJudgeResponse(BaseModel):
    """Structured scoring response returned by the LLM judge."""

    model_config = ConfigDict(extra="forbid")

    clarification_quality: int = Field(ge=1, le=5)
    assumption_control: int = Field(ge=1, le=5)
    reasoning_relevance_constraint_alignment: int = Field(ge=1, le=5)
    clarification_quality_reason: str | None = None
    assumption_control_reason: str | None = None
    reasoning_relevance_constraint_alignment_reason: str | None = None


class RegressionJudgeOutcome(BaseModel):
    """Final judge outcome stored on the regression result."""

    model_config = ConfigDict(extra="forbid")

    judge_status: str
    judge_error: str | None = None
    llm_called: bool
    request: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
