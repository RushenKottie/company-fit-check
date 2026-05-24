"""Typed request/response models for the stateless user simulator."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StartCaseRequest(BaseModel):
    """Request for the initial prompt and PDF path of one case."""

    model_config = ConfigDict(extra="forbid")

    case_id: int


class StartCaseResponse(BaseModel):
    """Initial-turn payload returned to the runner."""

    model_config = ConfigDict(extra="forbid")

    case_id: int
    case_name: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    pdf_path: str = Field(min_length=1)


class ReplyToAgentRequest(BaseModel):
    """Request for a simulated user reply to the latest agent message."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    case_id: int
    agent_message: str = Field(min_length=1)


class ReplyToAgentResponse(BaseModel):
    """Plain-text user reply returned to the runner."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    case_id: int
    case_name: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class UserSimulatorError(RuntimeError):
    """Typed simulator error returned to the runner layer."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
