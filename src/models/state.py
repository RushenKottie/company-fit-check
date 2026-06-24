"""LangGraph-compatible state for the current workflow."""

from typing import Literal, TypedDict

from pydantic import BaseModel, Field, field_serializer

from models.input import UserInput


HardFilterScope = Literal["result_set", "company_candidate", "unknown"]
HardFilterOperator = Literal[
    "exactly",
    "at_least",
    "at_most",
    "greater_than",
    "less_than",
    "between",
    "before",
    "after",
    "unknown",
]


class HardFilter(BaseModel):
    """A concrete mandatory filter with a literal numeric/date/money value."""

    text: str = Field(
        description="Original or lightly normalized user requirement text."
    )
    scope: HardFilterScope = Field(
        default="unknown",
        description="Whether the filter applies to the result set or each company.",
    )
    operator: HardFilterOperator = Field(
        default="unknown",
        description="Comparison operator expressed by the user.",
    )
    value: str = Field(
        default="",
        description="Normalized literal value from the requirement.",
    )
    minimum: int | None = Field(
        default=None,
        description="Numeric lower bound when the requirement has one.",
    )
    maximum: int | None = Field(
        default=None,
        description="Numeric upper bound when the requirement has one.",
    )


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
    hard_filters: list[HardFilter] = Field(default_factory=list)
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
    location: str = Field(
        default="",
        description="Company headquarters or primary location in City, Country format.",
    )
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


class FinalAxisResult(BaseModel):
    """Final per-axis score prepared for user-facing exports."""

    axis_name: str
    score: float


class FinalCompanyResult(BaseModel):
    """Final company fit result prepared for user-facing exports."""

    company_name: str
    website_or_linkedin: str
    location: str
    industry: str
    company_size: str
    discovery_reason: str
    overall_score: float
    axis_scores: list[FinalAxisResult] = Field(default_factory=list)

    @field_serializer("axis_scores", when_used="json")
    def serialize_axis_scores(self, axis_scores: list[FinalAxisResult]) -> str:
        """Render axis scores as one readable CSV cell."""

        return "; ".join(
            f"{axis.axis_name}: {axis.score:.1f}%"
            for axis in axis_scores
        )


SESSION_STATUS_RUNNING = "running"
SESSION_STATUS_NEEDS_CLARIFICATION = "needs_clarification"
SESSION_STATUS_COMPLETED = "completed"
SESSION_STATUS_FAILED = "failed"
TERMINAL_SESSION_STATUSES = {SESSION_STATUS_COMPLETED, SESSION_STATUS_FAILED}
INTERRUPTED_SESSION_STATUSES = {
    SESSION_STATUS_FAILED,
    SESSION_STATUS_NEEDS_CLARIFICATION,
}
PII_MASKING_STATUS_NOT_STARTED = "not_started"
PII_MASKING_STATUS_PASSED = "passed"
PII_MASKING_STATUS_FAILED = "failed"

SessionStatus = Literal[
    "running",
    "needs_clarification",
    "completed",
    "failed",
]

ClarificationTarget = Literal["user_input_interpretation"]
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
    final_results: list[FinalCompanyResult]
    pending_clarification_message: str | None
    latest_clarification_response: str | None
    clarification_target: ClarificationTarget | None
    latest_user_message_text: str | None
    latest_user_message_kind: UserMessageKind | None
    guardrail_rephrase_source: UserMessageKind | None
    run_id: str | None
    user_input_interpretation_clarification_iterations: int
    session_status: SessionStatus
    error: str | None
