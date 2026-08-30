"""Unified competition run-log contract and central iteration logger.

Implements Competition Requirement 5 (Run-Log Requirements) / Deliverable
#3 (Run & Iteration Logs): every candidate evaluation, model merge,
ablation study, inner refinement patch, ensemble round, and final
production run is recorded as a structured ``IterationLogEntry`` appended
to ``runs/<run_id>/iteration_logs.jsonl`` through the thread-safe
``CentralIterationLogger``.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class IterationLogEntry(BaseModel):
    """Structured JSON record for a single pipeline iteration.

    Attributes:
        iteration_id: Unique human-readable iteration tag (e.g. ``cand_1``,
            ``merge_1``, ``t0_k0``, ``ens_r0``, ``final_prod``).
        stage: Pipeline stage name (``INITIALIZATION``, ``REFINEMENT``,
            ``ENSEMBLING``, or ``FINALIZATION``).
        hypothesis: What the agent intended to try and why (Requirement 1).
        code_diff: Unified diff of the exact code modification applied
            (Requirement 2).
        metrics: Resulting metrics dictionary (Requirement 3).
        validation_score: Primary evaluation score.
        delta_from_baseline: Signed delta relative to the baseline anchor.
        error_recovery_events: Syntax errors, exceptions, timeouts,
            guardrail repairs, or debugger recovery rounds (Requirement 4).
        success: Whether the iteration produced an executable, scored
            artifact.
        target_component: Targeted module/component name.
        branch_index: Parallel branch seed index.
        timestamp: ISO-8601 timestamp.
        duration_seconds: Execution runtime in seconds.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    iteration_id: str = Field(description="Unique human-readable iteration tag.")
    stage: str = Field(description="Pipeline stage name.")
    hypothesis: str = Field(
        validation_alias=AliasChoices("hypothesis", "plan"),
        description="What the agent intended to try and why.",
    )
    code_diff: str = Field(default="", description="Unified diff applied.")
    metrics: dict[str, float] = Field(default_factory=dict, description="Metrics dictionary.")
    validation_score: float | None = Field(default=None, description="Primary evaluation score.")
    delta_from_baseline: float | None = Field(default=None, description="Delta over baseline.")
    error_recovery_events: list[str] = Field(
        validation_alias=AliasChoices("error_recovery_events", "errors"),
        default_factory=list,
        description="Error and recovery events.",
    )
    success: bool = Field(description="Whether the iteration produced a scored artifact.")
    target_component: str | None = Field(default=None, description="Targeted component.")
    branch_index: int | None = Field(default=None, description="Parallel branch index.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now().astimezone(),
        description="ISO-8601 timestamp.",
    )
    duration_seconds: float | None = Field(default=None, description="Execution runtime.")

    @property
    def plan(self) -> str:
        """Backward-compatible alias for ``hypothesis``."""
        return self.hypothesis

    @property
    def errors(self) -> list[str]:
        """Backward-compatible alias for ``error_recovery_events``."""
        return self.error_recovery_events


class CentralIterationLogger:
    """Thread-safe JSONL writer for the unified iteration log.

    Attributes:
        logs_path: Absolute path of the ``iteration_logs.jsonl`` file.
    """

    def __init__(self, logs_path: str | Path) -> None:
        """Create a logger writing to ``logs_path``.

        Args:
            logs_path: Path of the ``iteration_logs.jsonl`` file; parent
                directories are created on demand.
        """
        self.logs_path = Path(logs_path).resolve()
        self.logs_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @classmethod
    def for_run(cls, runs_dir: str | Path, run_id: str) -> CentralIterationLogger:
        """Build a logger for ``runs/<run_id>/iteration_logs.jsonl``.

        Args:
            runs_dir: Root directory holding per-run sandboxes.
            run_id: Identifier of the current run.

        Returns:
            A logger appending to the run's unified iteration log.
        """
        return cls(Path(runs_dir) / run_id / "iteration_logs.jsonl")

    def append(self, entry: IterationLogEntry) -> None:
        """Append one entry as a JSON line with an immediate flush.

        Thread-safe: concurrent appends are serialized by an internal lock,
        and every line is flushed before the writer returns.
        """
        line = entry.model_dump_json() + "\n"
        with self._lock, self.logs_path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(line)
            handle.flush()

    def read_all(self) -> list[IterationLogEntry]:
        """Return every parsed entry currently in the log file.

        Returns:
            An empty list when the log file does not exist.
        """
        if not self.logs_path.exists():
            return []
        entries: list[IterationLogEntry] = []
        for line in self.logs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(IterationLogEntry.model_validate_json(line))
        return entries
