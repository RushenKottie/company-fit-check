"""Input model for the current workflow."""

from pydantic import BaseModel, Field


class UserInput(BaseModel):
    """PDF CV bytes plus the user's prompt."""

    cv_pdf_bytes: bytes = Field(repr=False)
    prompt: str
