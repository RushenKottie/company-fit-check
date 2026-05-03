"""LangGraph-compatible state for the current workflow."""

from typing import Literal, TypedDict

from pydantic import BaseModel, Field

from company_fit_check.models.input import UserInput


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


class PromptInterpretation(BaseModel):
    """Structured interpretation of the prompt and simplified CV."""

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
    axis_scores: list[AxisScore] = Field(default_factory=list)
    overall_score: float = Field(ge=0.0, le=100.0)


SessionStatus = Literal[
    "running",
    "needs_clarification",
    "completed",
    "failed",
]

ClarificationTarget = Literal["interpretation", "company_search"]


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
    debug_session_id: str | None
    mlflow_run_id: str | None
    interpretation_clarification_iterations: int
    company_search_clarification_iterations: int
    session_status: SessionStatus
    error: str | None
