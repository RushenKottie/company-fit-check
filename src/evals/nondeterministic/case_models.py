"""Data models for non-deterministic evaluation case files."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CommunicationStyle(BaseModel):
    """Communication-style definition for one future simulator case."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    behavioral_traits: list[str] = Field(min_length=1)


class NondeterministicCase(BaseModel):
    """One flat non-deterministic case definition."""

    model_config = ConfigDict(extra="forbid")

    id: int
    name: str = Field(min_length=1)
    profession: str = Field(min_length=1)
    experience: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    pdf_path: str = Field(min_length=1)
    filter_criteria: list[str] = Field(min_length=1)
    axes: list[str] = Field(min_length=1)
    communication_style: CommunicationStyle
    first_prompt: str = Field(min_length=1)
