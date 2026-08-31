"""Search retrieval contract schemas.

These models capture the structured output of the web-search retriever
agent: candidate model cards with rationale and cleaned example code, plus
a container for a full retrieval response.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from problem_2_v2.contracts.code_utils import extract_python_code


class ModelCard(BaseModel):
    """Structured description of a web-retrieved candidate model.

    Attributes:
        model_name: Name of the candidate model or architecture.
        rationale: Why this model is relevant to the task.
        example_code: Cleaned example Python snippet (fences stripped).
        library_dependencies: Libraries required to run the example code.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model_name: str = Field(min_length=1, description="Candidate model name.")
    rationale: str = Field(description="Relevance rationale for the task.")
    example_code: str = Field(description="Cleaned example Python snippet.")
    library_dependencies: list[str] = Field(
        default_factory=list,
        description="Required libraries.",
    )

    @field_validator("example_code", mode="before")
    @classmethod
    def _clean_example_code(cls, value: object) -> object:
        """Strip markdown fences from raw LLM code output.

        Fenced blocks are extracted; raw code (even if only a fragment)
        is preserved verbatim.
        """
        if isinstance(value, str):
            cleaned = extract_python_code(value)
            if cleaned or not value.strip():
                return cleaned
        return value


class RetrievedCandidates(BaseModel):
    """Container for a full retrieval response.

    Attributes:
        candidates: The list of candidate model cards.
        total_found: Total number of candidates found.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    candidates: list[ModelCard] = Field(default_factory=list, description="Candidate model cards.")
    total_found: int = Field(description="Total candidates found.")

