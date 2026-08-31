"""Final artifact producer agent ($A_finalizer$).

Turns the winning ensemble solution into a production-ready artifact:
subsampling constraints are stripped so the model trains on the complete
dataset, model serialization and a ``metrics.json`` export are injected,
and the production script is executed with an extended timeout and the
debugger fallback, yielding a ``FinalArtifact`` describing ``./final/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import logfire
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent

from problem_2_v2.console import announce, format_score
from problem_2_v2.contracts.code_utils import (
    compute_code_diff,
    extract_python_code,
    validate_python_syntax,
)
from problem_2_v2.contracts.iteration import (
    CentralIterationLogger,
    IterationLogEntry,
    branch_index_from_run_id,
    declared_baseline,
)
from problem_2_v2.contracts.task import TaskSpecification
from problem_2_v2.execution.pipeline import ExecutionConfig
from problem_2_v2.runner.debugger import DebuggerAgent
from problem_2_v2.runner.sandbox import SubprocessRunner

_FINALIZER_INSTRUCTIONS = (
    "You are a Kaggle grandmaster producing the final production artifact "
    "for a completed machine learning competition.\n"
    "# Your task\n"
    "- Identify and remove all subsampling or row-capping constraints from "
    "the winning solution (e.g. `.head(30000)`, `.sample(n=30000)`, or "
    "`[:30000]` slicing of the training data).\n"
    "- Ensure the script trains on the complete dataset.\n"
    "- Add model serialization (e.g. `joblib.dump`, `torch.save`, or the "
    "framework-appropriate method) so the trained model is saved under "
    "./final/.\n"
    "- Add a JSON metrics export writing the final evaluation scores to "
    "./final/metrics.json.\n"
    "- Compute the final evaluation metrics with the official harness "
    "('from evaluate import evaluate', then "
    "evaluate(val_user_ids, val_labels, val_predictions)) and report "
    "val_res['primary'] (or the task metric key) as the validation score.\n"
    "- Write the submission to ./final/submission.csv with the exact "
    "4-column schema 'row_id,user_id,video_id,score' required by submit.py "
    "(submit.HEADER).\n"
    "- Ensure row_id is a 0-indexed contiguous integer following the "
    "deterministic row order of data.load()['test'] from the baseline data "
    "module (read log_standard_4_08_to_4_21_pure.csv first, then "
    "log_standard_4_22_to_5_08_pure.csv, filter by date, preserving the "
    "original file order). user_id and video_id must be redundant columns "
    "strictly aligned to the test rows, and score must be a finite real "
    "number.\n"
    "- Print 'Final Validation Performance: {final_validation_score}' so "
    "the score can be parsed.\n"
    "# Response format\n"
    "- Respond with a single markdown code block (wrapped in ```) which is "
    "the full production script.\n"
    "- There should be no additional headings or text in your response."
)

_MODEL_EXTENSIONS = frozenset({".pkl", ".joblib", ".pt", ".pth", ".onnx", ".h5", ".hdf5", ".sav"})


class FinalArtifact(BaseModel):
    """Production-ready artifact produced by the finalizer.

    Attributes:
        code: Final production script (possibly repaired by the debugger).
        output_dir: Path of the ``./final/`` output directory.
        model_paths: Serialized model file paths found in the output dir.
        metrics: Parsed final evaluation scores from ``metrics.json``.
        submission_path: Path of the submission CSV, if produced.
        validation_score: Final validation score, if the run succeeded.
        success: Whether the production run produced a validation score.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    code: str = Field(description="Final production script.")
    output_dir: str = Field(description="Output directory holding final artifacts.")
    model_paths: list[str] = Field(default_factory=list, description="Serialized model files.")
    metrics: dict[str, float] = Field(default_factory=dict, description="Final evaluation metrics.")
    submission_path: str | None = Field(default=None, description="Submission CSV path.")
    validation_score: float | None = Field(default=None, description="Final validation score.")
    success: bool = Field(description="Whether the production run produced a score.")


class FinalArtifactProducer:
    """Finalizes the winning solution into a production artifact.

    Attributes:
        agent: Pydantic AI agent rewriting the script for full-data training.
        debugger: Debugger agent with an extended production timeout.
        config: Execution configuration.
    """

    def __init__(
        self,
        debugger: DebuggerAgent | None = None,
        model: str = "openai:gpt-4o",
        config: ExecutionConfig | None = None,
    ) -> None:
        """Create the final artifact producer.

        Args:
            debugger: Debugger agent used for production execution; built
                with the extended production timeout when omitted.
            model: Pydantic AI model string.
            config: Execution configuration.
        """
        self.config = config or ExecutionConfig()
        self.agent = Agent(
            model,
            name="finalizer_agent",
            output_type=str,
            instructions=_FINALIZER_INSTRUCTIONS,
            defer_model_check=True,
        )
        self.debugger = debugger or DebuggerAgent(
            runner=SubprocessRunner(
                runs_dir=self.config.sandbox_base_dir,
                timeout_seconds=self.config.production_timeout_seconds,
            ),
            model=model,
            max_debug_rounds=self.config.max_debug_rounds,
        )

    def produce(
        self, code: str, spec: TaskSpecification, run_id: str = "finalize"
    ) -> FinalArtifact:
        """Produce the final artifact from the winning solution.

        Args:
            code: The winning solution script ($s^*_{\\text{ens}}$).
            spec: The task specification.
            run_id: Identifier of the finalization run.

        Returns:
            A ``FinalArtifact`` describing the production output.
        """
        prompt = self.build_prompt(code, spec)
        with logfire.span("finalizer.generate", run_id=run_id):
            response = self.agent.run_sync(prompt)
        rewritten = extract_python_code(response.output)

        if not rewritten:
            logfire.warn("finalizer.no_code", run_id=run_id)
            return FinalArtifact(
                code=code,
                output_dir="",
                model_paths=[],
                metrics={},
                submission_path=None,
                validation_score=None,
                success=False,
            )

        valid, error = validate_python_syntax(rewritten)
        if not valid:
            logfire.warn("finalizer.invalid_syntax", run_id=run_id, error=error)

        announce("[Finalizer] Stripping subsampling and training on complete dataset...")
        with logfire.span("finalizer.production_exec", run_id=run_id):
            outcome = self.debugger.debug(
                rewritten,
                run_id=run_id,
                candidate_id="final",
                dataset_dir=spec.dataset_dir,
                dataset_files=spec.dataset_files,
            )
        result = outcome.result

        final_dir = Path(self.debugger.runner.runs_dir) / run_id / "sandbox_final" / "final"
        announce(
            f"[Finalizer] Production run complete. Score: {format_score(result.validation_score)}"
        )

        model_paths = self._model_paths(final_dir)
        metrics = self._load_metrics(final_dir / "metrics.json")
        submission_path = self._submission_path(final_dir)
        errors: list[str] = []
        if outcome.debug_rounds > 0:
            errors.append(f"debugger applied {outcome.debug_rounds} repair round(s)")
        if result.stderr:
            errors.append(result.stderr[-500:])
        anchor = declared_baseline(spec.baseline_score)
        CentralIterationLogger.for_run(self.debugger.runner.runs_dir, run_id).append(
            IterationLogEntry(
                iteration_id="final_prod",
                stage="FINALIZATION",
                hypothesis=(
                    "Full-dataset training and production artifact generation. "
                    f"Serialized models: {', '.join(model_paths) or 'none'}. "
                    f"Submission: {submission_path or 'none'}."
                ),
                code_diff=compute_code_diff(code, outcome.code),
                metrics=metrics,
                validation_score=result.validation_score,
                delta_from_baseline=(
                    spec.metric_direction.delta(result.validation_score, anchor)
                    if result.validation_score is not None and anchor is not None
                    else None
                ),
                error_recovery_events=errors,
                success=result.validation_score is not None,
                target_component="FINAL_PRODUCTION",
                branch_index=branch_index_from_run_id(run_id),
                duration_seconds=result.duration_seconds,
            )
        )
        return FinalArtifact(
            code=outcome.code,
            output_dir=str(final_dir),
            model_paths=model_paths,
            metrics=metrics,
            submission_path=submission_path,
            validation_score=result.validation_score,
            success=result.validation_score is not None,
        )

    @staticmethod
    def build_prompt(code: str, spec: TaskSpecification) -> str:
        """Build the finalizer prompt from the winning solution and task.

        Args:
            code: The winning solution script.
            spec: The task specification.

        Returns:
            The finalization prompt.
        """
        task_desc = (
            spec.raw_description
            if spec.raw_description
            else (
                f"{spec.task_name}\n{spec.description}\nEvaluation metric: {spec.metric_name}"
            ).strip()
        )

        return (
            f"# Introduction\nYou are a Kaggle grandmaster producing the "
            f"final production artifact for a completed competition.\n"
            f"# Winning Solution\n{code}\n"
            f"# Task Description\n{task_desc}\n"
            f"# Your task\n"
            f"- Remove all subsampling or row-capping constraints (e.g. "
            f".head(30000), .sample(n=30000), or [:30000] slicing of the "
            f"training data).\n"
            f"- Ensure the script trains on the complete dataset.\n"
            f"- Add model serialization (e.g. joblib.dump, torch.save, or "
            f"the framework-appropriate method) saving the trained model "
            f"under ./final/.\n"
            f"- Add a JSON metrics export writing the final evaluation "
            f"scores to ./final/metrics.json.\n"
            f"- Compute the final evaluation metrics with the official "
            f"harness ('from evaluate import evaluate', then "
            f"evaluate(val_user_ids, val_labels, val_predictions)) and "
            f"report val_res['primary'] (or the task metric key) as the "
            f"validation score.\n"
            f"- Write the submission to ./final/submission.csv with the "
            f"exact 4-column schema 'row_id,user_id,video_id,score' "
            f"required by submit.py (submit.HEADER).\n"
            f"- Ensure row_id is a 0-indexed contiguous integer following "
            f"the deterministic row order of data.load()['test'] (read "
            f"log_standard_4_08_to_4_21_pure.csv first, then "
            f"log_standard_4_22_to_5_08_pure.csv, filter by date, "
            f"preserving the original file order). user_id and video_id "
            f"must be redundant columns strictly aligned to the test rows, "
            f"and score must be a finite real number.\n"
            f"- Print 'Final Validation Performance: "
            f"{{final_validation_score}}' for parsing.\n"
            f"- Respond with a single markdown code block containing the "
            f"full production script."
        )

    @staticmethod
    def _model_paths(final_dir: Path) -> list[str]:
        """List serialized model files found in the output directory."""
        if not final_dir.exists():
            return []
        return [
            str(path)
            for path in sorted(final_dir.iterdir())
            if path.is_file() and path.suffix.lower() in _MODEL_EXTENSIONS
        ]

    @staticmethod
    def _submission_path(final_dir: Path) -> str | None:
        """Return the submission CSV path when present."""
        submission = final_dir / "submission.csv"
        return str(submission) if submission.exists() else None

    @staticmethod
    def _load_metrics(metrics_path: Path) -> dict[str, float]:
        """Parse ``metrics.json`` into a flat numeric metrics dictionary.

        Non-numeric entries are skipped so the metrics dict stays numeric.
        """
        if not metrics_path.exists():
            return {}
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(data, dict):
            return {}
        metrics: dict[str, float] = {}
        for key, value in data.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
            elif isinstance(value, str):
                try:
                    metrics[key] = float(value)
                except ValueError:
                    continue
        return metrics
