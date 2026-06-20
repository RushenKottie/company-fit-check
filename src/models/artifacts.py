"""Reusable CSV artifact models."""

from pydantic import BaseModel, Field


class GeneratedCsvArtifact(BaseModel):
    """CSV file produced by the backend for downstream interfaces."""

    filename: str
    content_type: str
    content_bytes: bytes = Field(repr=False)


GeneratedArtifact = GeneratedCsvArtifact
