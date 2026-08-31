"""Iterative ensemble optimization pipeline (Algorithm 3).

Coordinates the ensemble planner and ensembler across $R$ rounds,
evaluating each merged script in the sandbox, streaming structured
iteration logs, and selecting the optimal solution $s^*_{ens}$ across all
ensemble scripts and the individual candidates (never degrading below the
best single candidate).
"""

from __future__ import annotations

import logfire
from pydantic import BaseModel, ConfigDict, Field

from problem_2_v2.console import announce, format_delta, format_score
from problem_2_v2.contracts.code_utils import compute_code_diff
from problem_2_v2.contracts.enums import MetricDirection
from problem_2_v2.contracts.guardrails import EnsembleStrategy
from problem_2_v2.contracts.iteration import (
    CentralIterationLogger,
    IterationLogEntry,
    branch_index_from_run_id,
)
from problem_2_v2.contracts.task import PipelineArtifact, TaskSpecification
from problem_2_v2.ensembling.ensembler import EnsemblerAgent
from problem_2_v2.ensembling.planner import EnsemblePlannerAgent
from problem_2_v2.execution.pipeline import ExecutionGuardrailPipeline
from problem_2_v2.runner.sandbox import SubprocessRunner


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
        execution: Unified execution guardrail pipeline used by the
            ensembler, if any.
    """

    def __init__(
        self,
        planner: EnsemblePlannerAgent,
        ensembler: EnsemblerAgent,
        runner: SubprocessRunner,
        rounds: int = 3,
        execution: ExecutionGuardrailPipeline | None = None,
    ) -> None:
        """Create the ensemble pipeline.

        Args:
            planner: Ensemble planning agent.
            ensembler: Code ensembling agent.
            runner: Sandbox runner.
            rounds: Number of ensemble rounds (R).
            execution: Unified execution guardrail pipeline; wired into
                the ensembler when it has none.
        """
        self.planner = planner
        self.ensembler = ensembler
        self.runner = runner
        self.rounds = rounds
        self.execution = execution
        if execution is not None and ensembler.execution is None:
            ensembler.execution = execution

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

        Raises:
            ValueError: When no candidate solutions are provided.
        """
        if not solutions:
            raise ValueError("No candidate solutions provided for ensembling.")

        best_individual = max(solutions, key=lambda a: a.validation_score or float("-inf"))
        if len(solutions) == 1 or self.rounds <= 0:
            return EnsembleResult(
                best_artifact=best_individual,
                best_score=best_individual.validation_score,
                best_code=best_individual.full_code,
                best_submission_path=None,
                logs_path=None,
                rounds_executed=0,
            )

        logger = CentralIterationLogger.for_run(self.runner.runs_dir, run_id)
        branch_index = branch_index_from_run_id(run_id)
        direction = spec.metric_direction

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
                        logger.append(
                            IterationLogEntry(
                                iteration_id=f"ens_r{r}",
                                stage="ENSEMBLING",
                                hypothesis=strategy.natural_language_plan,
                                code_diff="",
                                metrics={},
                                validation_score=None,
                                delta_from_baseline=None,
                                error_recovery_events=[str(exc)],
                                success=False,
                                target_component=f"ENSEMBLE_{strategy.method.value}",
                                branch_index=branch_index,
                                duration_seconds=None,
                            )
                        )
                        continue
                attempts.append((strategy, run.score))
                rounds_executed = r + 1
                round_delta = (
                    direction.delta(run.score, baseline)
                    if run.score is not None and baseline is not None
                    else None
                )
                announce(
                    f"[Ensemble Round {r + 1}/{self.rounds}] Strategy: "
                    f"'{strategy.method.value}' -> "
                    f"Score: {format_score(run.score)} (Δ {format_delta(round_delta)})"
                )
                logger.append(
                    IterationLogEntry(
                        iteration_id=f"ens_r{r}",
                        stage="ENSEMBLING",
                        hypothesis=strategy.natural_language_plan,
                        code_diff=compute_code_diff(best_individual.full_code, run.code),
                        metrics=({"primary": run.score} if run.score is not None else {}),
                        validation_score=run.score,
                        delta_from_baseline=round_delta,
                        error_recovery_events=(
                            [run.result.stderr[-500:]]
                            if not run.success and run.result is not None and run.result.stderr
                            else []
                        ),
                        success=run.success,
                        target_component=f"ENSEMBLE_{strategy.method.value}",
                        branch_index=branch_index,
                        duration_seconds=(
                            run.result.duration_seconds if run.result is not None else None
                        ),
                    )
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
            logs_path=str(logger.logs_path),
            rounds_executed=rounds_executed,
        )

    @staticmethod
    def _accepts(
        candidate: float | None,
        best: float | None,
        direction: MetricDirection,
    ) -> bool:
        """Return whether a candidate strictly outperforms the best score."""
        if candidate is None:
            return False
        if best is None:
            return True
        return direction.is_better(candidate, best)
