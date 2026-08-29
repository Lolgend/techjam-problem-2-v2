"""Master configuration for the MLE-STAR pipeline.

Defines every pipeline hyperparameter as a validated Pydantic model so the
CLI and Python API share a single source of truth.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SearchProviderName = Literal["tavily", "google", "duckduckgo", "mock"]


class MLEStarConfig(BaseModel):
    """Hyperparameter configuration for the full MLE-STAR pipeline.

    Attributes:
        model: Pydantic AI model string used by all LLM agents.
        search_provider: Web search backend name.
        num_candidates: Number of candidate models to retrieve (M).
        num_branches: Number of parallel initial + refine branches (L).
        outer_loops: Number of outer refinement iterations (T).
        inner_loops: Number of inner refinement iterations (K).
        ensemble_rounds: Number of ensemble rounds (R).
        seeds: Explicit random seeds per branch; defaults to
            ``range(num_branches)`` when ``None`` (e.g. ``[42, 123]``).
        subsample_size: Maximum training rows for fast experimentation.
        timeout_seconds: Per-script sandbox wall-clock timeout.
        production_timeout_seconds: Extended timeout for full-data runs.
        max_debug_rounds: Debugger repair budget.
        runs_dir: Root directory holding per-run sandboxes.
        final_output_dir: Production output directory (default ``final``).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: str = Field(default="openai:gpt-4o", description="LLM model identifier.")
    search_provider: SearchProviderName = Field(default="duckduckgo", description="Search backend.")
    num_candidates: int = Field(default=4, gt=0, description="Candidate models (M).")
    num_branches: int = Field(default=2, gt=0, description="Parallel branches (L).")
    outer_loops: int = Field(default=3, gt=0, description="Outer iterations (T).")
    inner_loops: int = Field(default=3, gt=0, description="Inner iterations (K).")
    ensemble_rounds: int = Field(default=3, gt=0, description="Ensemble rounds (R).")
    seeds: list[int] | None = Field(
        default=None,
        description="Per-branch random seeds (e.g. [42, 123]); None means range(num_branches).",
    )
    subsample_size: int = Field(default=30000, gt=0, description="Fast-experiment row cap.")
    timeout_seconds: int = Field(default=600, gt=0, description="Sandbox timeout in seconds.")
    production_timeout_seconds: int = Field(
        default=3600, gt=0, description="Production timeout in seconds."
    )
    max_debug_rounds: int = Field(default=3, ge=0, description="Debugger repair budget.")
    runs_dir: str = Field(default="runs", description="Sandbox root directory.")
    final_output_dir: str = Field(default="final", description="Production output directory.")
