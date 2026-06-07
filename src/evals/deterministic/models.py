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
