"""Iterative ensemble optimization pipeline (Algorithm 3).

Coordinates the ensemble planner and ensembler across $R$ rounds,
evaluating each merged script in the sandbox, streaming structured
iteration logs, and selecting the optimal solution $s^*_{ens}$ across all
ensemble scripts and the individual candidates (never degrading below the
best single candidate).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import logfire
from pydantic import BaseModel, ConfigDict, Field

from problem_2_v2.contracts.enums import MetricDirection
from problem_2_v2.contracts.guardrails import EnsembleStrategy
from problem_2_v2.contracts.task import PipelineArtifact, TaskSpecification
from problem_2_v2.ensembling.ensembler import EnsemblerAgent
from problem_2_v2.ensembling.planner import EnsemblePlannerAgent
from problem_2_v2.runner.sandbox import SubprocessRunner


class EnsembleIterationLogRecord(BaseModel):
    """Structured log record streamed per ensemble round.

    Attributes:
        round_index: Ensemble round index (r).
        method: Ensembling method applied.
        plan: Natural-language plan text.
        validation_score: Score of the merged script, if any.
        delta_from_baseline: Signed delta over the best individual score.
        success: Whether the round produced a score.
        errors: Error events encountered.
        timestamp: When the round finished.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    round_index: int = Field(description="Ensemble round index.")
    method: str = Field(description="Ensembling method.")
    plan: str = Field(description="Plan text.")
    validation_score: float | None = Field(default=None, description="Merged score.")
    delta_from_baseline: float | None = Field(default=None, description="Delta over baseline.")
    success: bool = Field(description="Whether the round produced a score.")
    errors: list[str] = Field(default_factory=list, description="Error events.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now().astimezone())


class EnsembleResult(BaseModel):
    """Final outcome of the ensemble optimization run.

    Attributes:
        best_artifact: The optimal solution artifact $s^*_{ens}$.
        best_score: The optimal validation score.
        best_code: The optimal solution script.
        best_submission_path: Submission file of the optimal script, if any.
        logs_path: Path of the streamed iteration log file.
        rounds_executed: Number of ensemble rounds executed.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    best_artifact: PipelineArtifact = Field(description="Optimal solution artifact.")
    best_score: float | None = Field(default=None, description="Optimal validation score.")
    best_code: str = Field(description="Optimal solution script.")
    best_submission_path: str | None = Field(default=None, description="Optimal submission file.")
    logs_path: str | None = Field(default=None, description="Iteration log file path.")
    rounds_executed: int = Field(default=0, description="Rounds executed.")


class EnsemblePipeline:
    """Orchestrates the iterative ensemble search (Algorithm 3).

    Attributes:
        planner: Ensemble planning agent.
        ensembler: Code ensembling agent.
        runner: Sandbox runner (provides the runs directory).
        rounds: Number of ensemble rounds (R).
    """

    def __init__(
        self,
        planner: EnsemblePlannerAgent,
        ensembler: EnsemblerAgent,
        runner: SubprocessRunner,
        rounds: int = 3,
    ) -> None:
        """Create the ensemble pipeline.

        Args:
            planner: Ensemble planning agent.
            ensembler: Code ensembling agent.
            runner: Sandbox runner.
            rounds: Number of ensemble rounds (R).
        """
        self.planner = planner
        self.ensembler = ensembler
        self.runner = runner
        self.rounds = rounds

    def run(
        self,
        spec: TaskSpecification,
        solutions: list[PipelineArtifact],
        run_id: str = "ensembling",
    ) -> EnsembleResult:
        """Run the iterative ensemble optimization.

        Args:
            spec: The task specification.
            solutions: The candidate solution artifacts.
            run_id: Identifier of the current run.

        Returns:
            An ``EnsembleResult`` with the optimal solution artifact.
        """
        logs_path = Path(self.runner.runs_dir) / run_id / "iteration_logs.jsonl"
        logs_path.parent.mkdir(parents=True, exist_ok=True)
        direction = spec.metric_direction

        best_individual = max(solutions, key=lambda a: a.validation_score or float("-inf"))
        baseline = best_individual.validation_score
        best_code = best_individual.full_code
        best_score = baseline
        best_artifact = best_individual
        best_submission: str | None = None

        attempts: list[tuple[EnsembleStrategy, float | None]] = []
        rounds_executed = 0

        with logfire.span("ensembling.run", run_id=run_id):
            for r in range(self.rounds):
                with logfire.span("ensembling.round", r=r):
                    if r == 0:
                        try:
                            strategy = self.planner.initial_plan(solutions)
                        except Exception as exc:
                            logfire.warn("ensembling.initial_plan.failed", error=str(exc))
                            break
                    else:
                        try:
                            strategy = self.planner.next_plan(solutions, attempts, r)
                        except Exception as exc:
                            logfire.warn("ensembling.next_plan.failed", r=r, error=str(exc))
                            break
                    try:
                        run = self.ensembler.ensemble(
                            spec, solutions, strategy, run_id=run_id, round_index=r
                        )
                    except Exception as exc:
                        logfire.warn("ensembling.ensembler.failed", r=r, error=str(exc))
                        self._append_log(
                            logs_path,
                            EnsembleIterationLogRecord(
                                round_index=r,
                                method=strategy.method.value,
                                plan=strategy.natural_language_plan,
                                validation_score=None,
                                delta_from_baseline=None,
                                success=False,
                                errors=[str(exc)],
                            ),
                        )
                        continue
                attempts.append((strategy, run.score))
                rounds_executed = r + 1
                self._append_log(
                    logs_path,
                    EnsembleIterationLogRecord(
                        round_index=r,
                        method=strategy.method.value,
                        plan=strategy.natural_language_plan,
                        validation_score=run.score,
                        delta_from_baseline=(
                            direction.delta(run.score, baseline)
                            if run.score is not None and baseline is not None
                            else None
                        ),
                        success=run.success,
                        errors=(
                            [run.result.stderr[-500:]]
                            if not run.success and run.result is not None and run.result.stderr
                            else []
                        ),
                    ),
                )
                if run.score is not None and self._accepts(run.score, best_score, direction):
                    best_score = run.score
                    best_code = run.code
                    best_submission = run.submission_path

        stage = (
            "ens_optimal"
            if best_code != best_individual.full_code
            else best_individual.iteration_stage
        )
        best_artifact = PipelineArtifact(
            version=0,
            full_code=best_code,
            validation_score=best_score,
            parent_version=None,
            applied_diff=None,
            iteration_stage=stage,
        )
        return EnsembleResult(
            best_artifact=best_artifact,
            best_score=best_score,
            best_code=best_code,
            best_submission_path=best_submission,
            logs_path=str(logs_path),
            rounds_executed=rounds_executed,
        )

    @staticmethod
    def _accepts(
        candidate: float | None,
        best: float | None,
        direction: MetricDirection,
    ) -> bool:
        """Return whether a candidate score is at least as good as best."""
        if candidate is None:
            return False
        if best is None:
            return True
        return candidate == best or direction.is_better(candidate, best)

    @staticmethod
    def _append_log(logs_path: Path, record: EnsembleIterationLogRecord) -> None:
        """Append a JSON line to the iteration log file."""
        with logs_path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
