"""Task specification, execution telemetry, and artifact lineage contracts.

These models form the foundation of the MLE-STAR ingestion pipeline: they
parse markdown problem descriptions into validated task specifications,
capture subprocess execution outcomes, and track the versioned lineage of
pipeline artifacts across refinement iterations.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from problem_2_v2.contracts.enums import MetricDirection, TaskType

_FIELD_LABEL_RE = re.compile(r"^\s*#{0,4}\s*\*\*(?P<label>[^*:]+):\*\*\s*(?P<value>.*)$")


class TaskSpecification(BaseModel):
    """Validated metadata describing an ML task for the MLE-STAR pipeline.

    Attributes:
        task_name: Human-readable competition or benchmark name.
        task_type: The family of the machine learning task.
        description: Free-form problem description.
        metric_name: Name of the evaluation metric (e.g. ``NDCG@10``).
        metric_direction: Whether higher or lower metric values are better.
        target_variable: The column the agent must predict.
        dataset_dir: Absolute path to the directory holding the dataset.
        dataset_files: Relative file names within ``dataset_dir``.
        baseline_score: The official baseline validation score to beat.
        constraints: Free-form execution or methodology constraints.
        subsample_size: Maximum training rows to use for fast experimentation.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_name: str = Field(default="", description="Name of the task or competition.")
    task_type: TaskType = Field(description="Machine learning task family.")
    description: str = Field(default="", description="Free-form problem description.")
    metric_name: str = Field(default="", description="Evaluation metric name.")
    metric_direction: MetricDirection = Field(
        default=MetricDirection.MAXIMIZE,
        description="Whether higher (MAXIMIZE) or lower (MINIMIZE) is better.",
    )
    target_variable: str = Field(default="", description="Prediction target column.")
    dataset_dir: str = Field(description="Directory containing the dataset files.")
    dataset_files: list[str] = Field(default_factory=list, description="Dataset file names.")
    baseline_score: float = Field(default=0.0, description="Official baseline validation score.")
    constraints: str = Field(default="", description="Free-form constraints.")
    subsample_size: int = Field(
        default=30000,
        description="Maximum training samples used for fast experimentation.",
    )

    @classmethod
    def from_markdown(cls, md_text: str, dataset_dir: str) -> TaskSpecification:
        """Parse a markdown problem description into a validated task spec.

        The parser recognises lines of the form ``**Field Name:** value``
        (optionally wrapped in markdown heading markers). Multi-line fields
        (``Description``, ``Constraints``) consume every following line
        until the next field label. Unknown labels are ignored so the
        parser tolerates extra prose in the problem description.

        Args:
            md_text: The raw markdown problem description.
            dataset_dir: Absolute path to the dataset directory.

        Returns:
            A fully validated ``TaskSpecification``.

        Raises:
            pydantic.ValidationError: If required fields are missing or
                invalid (e.g. an unknown ``Task Type`` value).
        """
        fields: dict[str, object] = {}
        current_multi_line: str | None = None
        multi_line_buffer: list[str] = []

        def flush_multi_line() -> None:
            if current_multi_line is not None:
                fields[current_multi_line] = "\n".join(multi_line_buffer).strip()
            multi_line_buffer.clear()

        for line in md_text.splitlines():
            match = _FIELD_LABEL_RE.match(line)
            if match is not None:
                flush_multi_line()
                label = match.group("label").strip().lower()
                value = match.group("value").strip()
                normalized = _normalize_label(label)
                if normalized in _MULTI_LINE_FIELDS:
                    current_multi_line = normalized
                    if value:
                        multi_line_buffer.append(value)
                else:
                    current_multi_line = None
                    _store_field(fields, normalized, value)
                continue
            if current_multi_line is not None:
                multi_line_buffer.append(line)

        flush_multi_line()

        fields["dataset_dir"] = dataset_dir
        return cls(**fields)


_MULTI_LINE_FIELDS = frozenset({"description", "constraints"})

_FIELD_MAP = {
    "task name": "task_name",
    "task type": "task_type",
    "metric name": "metric_name",
    "metric direction": "metric_direction",
    "target variable": "target_variable",
    "baseline score": "baseline_score",
    "dataset files": "dataset_files",
    "subsample size": "subsample_size",
}


def _normalize_label(label: str) -> str:
    """Normalize a parsed markdown label into a canonical lowercase key."""
    return re.sub(r"[^a-z0-9]+", " ", label).strip()


def _store_field(fields: dict[str, object], normalized_label: str, value: str) -> None:
    """Store a parsed single-line field value under its canonical key."""
    key = _FIELD_MAP.get(normalized_label)
    if key is None:
        return
    if key == "dataset_files":
        fields[key] = [item.strip() for item in re.split(r"[,;\s]+", value) if item.strip()]
    elif key == "subsample_size":
        fields[key] = int(value)
    elif key == "baseline_score":
        fields[key] = float(value)
    else:
        fields[key] = value


class ExecutionResult(BaseModel):
    """Captured outcome of a subprocess pipeline execution.

    Attributes:
        success: Whether the subprocess completed without a fatal error.
        stdout: Raw standard output of the subprocess.
        stderr: Raw standard error of the subprocess.
        returncode: Subprocess return code.
        duration_seconds: Wall-clock execution duration.
        validation_score: Parsed ``Final Validation Performance`` score.
        error_traceback: Captured traceback on failure, if any.
        gpu_memory_mb: Peak GPU memory usage, if measured.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    success: bool = Field(description="Whether execution succeeded.")
    stdout: str = Field(default="", description="Raw standard output.")
    stderr: str = Field(default="", description="Raw standard error.")
    returncode: int = Field(description="Subprocess return code.")
    duration_seconds: float = Field(description="Wall-clock execution duration.")
    validation_score: float | None = Field(
        default=None,
        description="Parsed 'Final Validation Performance' score.",
    )
    error_traceback: str | None = Field(
        default=None,
        description="Captured traceback on failure.",
    )
    gpu_memory_mb: float | None = Field(default=None, description="Peak GPU memory in MB.")

    _VALIDATION_SCORE_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"Final\s+Validation\s+Performance\s*:\s*(?P<score>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )

    def extract_validation_score(self, stdout: str) -> float | None:
        """Parse the ``Final Validation Performance: {score}`` line.

        Args:
            stdout: The subprocess standard output to scan.

        Returns:
            The parsed float score, or ``None`` when no match is found.
        """
        match = self._VALIDATION_SCORE_RE.search(stdout)
        if match is None:
            return None
        return float(match.group("score"))


class PipelineArtifact(BaseModel):
    """Versioned lineage record for a pipeline solution snapshot.

    Attributes:
        version: Monotonic version number of this artifact.
        full_code: The complete self-contained solution script.
        validation_score: The validation score achieved by this artifact.
        parent_version: Version of the parent artifact this one derives from.
        applied_diff: Unified diff applied to the parent to create this version.
        iteration_stage: Pipeline stage that produced this artifact.
        timestamp: When the artifact was created.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    version: int = Field(description="Monotonic artifact version number.")
    full_code: str = Field(description="The complete solution script.")
    validation_score: float = Field(description="Validation score achieved.")
    parent_version: int | None = Field(default=None, description="Parent artifact version.")
    applied_diff: str | None = Field(default=None, description="Unified diff from the parent.")
    iteration_stage: str = Field(description="Stage that produced this artifact.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp.",
    )
