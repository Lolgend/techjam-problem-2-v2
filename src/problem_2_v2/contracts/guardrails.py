"""Ensembling and guardrail contract schemas.

These models capture LLM-proposed ensembling strategies and the audit
outputs of the data leakage and data usage guardrails.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from problem_2_v2.contracts.enums import EnsembleMethod

__all__ = [
    "EnsembleMethod",
    "EnsembleStrategy",
    "DataLeakageStatus",
    "DataUsageStatus",
]

_YES_LEAK_RE = re.compile(r"^yes\s+data\s+leakage", re.IGNORECASE)
_NO_LEAK_RE = re.compile(r"^no\s+data\s+leakage", re.IGNORECASE)


class EnsembleStrategy(BaseModel):
    """LLM-proposed ensembling strategy across candidate solutions.

    Attributes:
        method: The ensembling technique to apply.
        natural_language_plan: Prose description of the ensembling plan.
        meta_learner_type: Meta-learner model class for stacking, if any.
        candidate_solution_ids: Identifiers of the solutions to ensemble.
        code_template: Optional code template implementing the plan.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    method: EnsembleMethod = Field(description="Ensembling technique.")
    natural_language_plan: str = Field(description="Ensembling plan in prose.")
    meta_learner_type: str | None = Field(
        default=None,
        description="Meta-learner model class for stacking.",
    )
    candidate_solution_ids: list[str] = Field(
        default_factory=list,
        description="Solutions to ensemble.",
    )
    code_template: str | None = Field(default=None, description="Optional code template.")


class DataLeakageStatus(BaseModel):
    """Result of the data leakage guardrail audit.

    Attributes:
        leakage_status: Raw detector output; accepts the paper-exact
            strings ``"Yes Data Leakage"`` and ``"No Data Leakage"``.
        is_leaking: Normalized boolean leakage flag.
        suspicious_code_block: The code block suspected of leaking.
        corrected_code_block: Auto-corrected block, if one was produced.
        explanation: Audit explanation in prose.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    leakage_status: str = Field(description="Raw leakage detector output.")
    is_leaking: bool = Field(description="Normalized leakage flag.")
    suspicious_code_block: str | None = Field(
        default=None,
        description="Code block suspected of leaking.",
    )
    corrected_code_block: str | None = Field(
        default=None,
        description="Auto-corrected block, if any.",
    )
    explanation: str = Field(description="Audit explanation.")

    @model_validator(mode="before")
    @classmethod
    def _normalize_leakage_status(cls, data: Any) -> Any:
        """Normalize paper-exact prompt strings into the boolean flag.

        When the detector returns ``"Yes Data Leakage"`` or ``"No Data
        Leakage"`` (case-insensitive, with optional trailing punctuation),
        ``is_leaking`` is derived from the string and overrides any
        explicitly provided value. Other status strings are left
        untouched.
        """
        if not isinstance(data, dict):
            return data
        status = data.get("leakage_status")
        if not isinstance(status, str):
            return data
        if _YES_LEAK_RE.match(status.strip()):
            data["is_leaking"] = True
        elif _NO_LEAK_RE.match(status.strip()):
            data["is_leaking"] = False
        return data


class DataUsageStatus(BaseModel):
    """Result of the data ingestion audit.

    Attributes:
        all_data_used: Whether every dataset source is consumed.
        missing_sources: Sources referenced by the task but unused by code.
        usage_recommendations: Recommended fixes in prose.
        improved_code_block: Improved ingestion code, if generated.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    all_data_used: bool = Field(description="Whether all data sources are used.")
    missing_sources: list[str] = Field(
        default_factory=list,
        description="Referenced but unused sources.",
    )
    usage_recommendations: str = Field(description="Recommended fixes.")
    improved_code_block: str | None = Field(
        default=None,
        description="Improved ingestion code, if any.",
    )
