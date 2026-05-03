"""Reusable generated artifact models."""

from pydantic import BaseModel, Field


class GeneratedArtifact(BaseModel):
    """Binary artifact produced by the backend for downstream interfaces."""

    filename: str
    content_type: str
    content_bytes: bytes = Field(repr=False)
