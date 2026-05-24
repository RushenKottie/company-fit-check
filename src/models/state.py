"""LangGraph-compatible state for the current workflow."""

from typing import Literal, TypedDict

from pydantic import BaseModel, Field

from models.input import UserInput


class CompanySearchCriteria(BaseModel):
    """Common company-search criteria plus an escape hatch."""

    locations: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    company_stages: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    work_modes: list[str] = Field(default_factory=list)
    must_have: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list)
    undefined: list[str] = Field(default_factory=list)


class Axis(BaseModel):
    """A user-perspective matching axis."""

    name: str
    description: str


class UserInputInterpretation(BaseModel):
    """Structured interpretation of the full user input and simplified CV."""

    company_search_criteria: CompanySearchCriteria
    axes: list[Axis]


class CompanyCandidate(BaseModel):
    """A discovered company candidate for later fit evaluation."""

    name: str
    website_or_linkedin: str
    industry: str
    company_size: str
    discovery_reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class AxisScore(BaseModel):
    """Score for one company on one matching axis."""

    axis: str
    percentage: float = Field(ge=0.0, le=100.0)


class CompanyScore(BaseModel):
    """Scored fit result for one discovered company."""

    company_name: str
    website_or_linkedin: str = ""
    axis_scores: list[AxisScore] = Field(default_factory=list)
    overall_score: float = Field(ge=0.0, le=100.0)


SessionStatus = Literal[
    "running",
    "needs_clarification",
    "completed",
    "failed",
]

ClarificationTarget = Literal["user_input_interpretation", "company_search"]
UserMessageKind = Literal["prompt", "clarification"]


class CompanyFitState(TypedDict, total=False):
    """Mutable in-memory state passed through the LangGraph workflow."""

    input: UserInput
    masked_cv_text: str | None
    pii_masking_status: Literal["not_started", "passed", "failed"]
    simplified_cv_text: str | None
    company_search_criteria: CompanySearchCriteria
    axes: list[Axis]
    companies: list[CompanyCandidate]
    company_scores: list[CompanyScore]
    pending_clarification_message: str | None
    latest_clarification_response: str | None
    clarification_target: ClarificationTarget | None
    latest_user_message_text: str | None
    latest_user_message_kind: UserMessageKind | None
    guardrail_rephrase_source: UserMessageKind | None
    run_id: str | None
    user_input_interpretation_clarification_iterations: int
    company_search_clarification_iterations: int
    session_status: SessionStatus
    error: str | None
