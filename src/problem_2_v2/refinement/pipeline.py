"""Nested refinement pipeline orchestrator (Algorithm 2).

Executes the outer exploration loop ($T$ iterations: ablate -> summarize
-> extract) and the inner refinement loop ($K$ iterations: plan -> code ->
patch -> guardrails -> evaluate), streaming structured iteration logs to
``runs/<run_id>/iteration_logs.jsonl`` and returning the best solution as
a finalized ``PipelineArtifact`` lineage.
"""

from __future__ import annotations

import logfire
from pydantic import BaseModel, ConfigDict, Field

from problem_2_v2.console import announce, format_delta, format_score
from problem_2_v2.contracts.code_utils import compute_code_diff
from problem_2_v2.contracts.enums import MetricDirection
from problem_2_v2.contracts.iteration import (
    CentralIterationLogger,
    IterationLogEntry,
    branch_index_from_run_id,
)
from problem_2_v2.contracts.refinement import RefinementPlan, TargetCodeBlock
from problem_2_v2.contracts.task import PipelineArtifact, TaskSpecification
from problem_2_v2.execution.pipeline import ExecutionConfig, ExecutionGuardrailPipeline
from problem_2_v2.guardrails.leakage import DataLeakageCheckerAgent
from problem_2_v2.guardrails.usage import DataUsageCheckerAgent
from problem_2_v2.refinement.ablation import AblationAgent, AblationSummarizerAgent
from problem_2_v2.refinement.coder import CoderAgent, patch_script
from problem_2_v2.refinement.extractor import CodeBlockExtractorAgent
from problem_2_v2.refinement.planner import RefinementPlannerAgent
from problem_2_v2.runner.debugger import DebuggerAgent
from problem_2_v2.runner.sandbox import SubprocessRunner


class RefinementResult(BaseModel):
    """Final outcome of the nested refinement run.

    Attributes:
        final_code: Best solution script found.
        final_score: Best validation score found.
        lineage: Versioned artifact lineage of improvements.
        logs_path: Path of the streamed iteration log file.
        outer_iterations: Number of outer loops executed.
        inner_iterations: Number of inner loops executed.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    final_code: str = Field(description="Best solution script.")
    final_score: float | None = Field(default=None, description="Best validation score.")
    lineage: list[PipelineArtifact] = Field(default_factory=list, description="Artifact lineage.")
    logs_path: str | None = Field(default=None, description="Iteration log file path.")
    outer_iterations: int = Field(default=0, description="Outer loops executed.")
    inner_iterations: int = Field(default=0, description="Inner loops executed.")


class RefinementPipeline:
    """Orchestrates the nested $T \\times K$ refinement loops.

    Attributes:
        ablation: Ablation generation agent.
        summarizer: Ablation summarization agent.
        extractor: Target code block extractor.
        planner: Adaptive refinement planner.
        coder: Code block refiner.
        execution: Unified execution guardrail pipeline (guardrails,
            sandbox execution, and the debugger loop).
        runner: Sandbox runner.
        outer_loops: Number of outer iterations (T).
        inner_loops: Number of inner iterations (K).
    """

    def __init__(
        self,
        ablation: AblationAgent,
        summarizer: AblationSummarizerAgent,
        extractor: CodeBlockExtractorAgent,
        planner: RefinementPlannerAgent,
        coder: CoderAgent,
        leakage: DataLeakageCheckerAgent | None = None,
        usage: DataUsageCheckerAgent | None = None,
        debugger: DebuggerAgent | None = None,
        runner: SubprocessRunner | None = None,
        execution: ExecutionGuardrailPipeline | None = None,
        outer_loops: int = 3,
        inner_loops: int = 3,
    ) -> None:
        """Create the refinement pipeline.

        Args:
            ablation: Ablation generation agent.
            summarizer: Ablation summarization agent.
            extractor: Target code block extractor.
            planner: Adaptive refinement planner.
            coder: Code block refiner.
            leakage: Data leakage guardrail.
            usage: Data usage guardrail.
            debugger: Execution debugger.
            runner: Sandbox runner.
            execution: Unified execution guardrail pipeline. When omitted,
                one is built from the leakage, usage, debugger, and runner
                components.
            outer_loops: Number of outer iterations (T).
            inner_loops: Number of inner iterations (K).

        Raises:
            ValueError: When neither ``execution`` nor the execution
                component quartet is provided.
        """
        self.ablation = ablation
        self.summarizer = summarizer
        self.extractor = extractor
        self.planner = planner
        self.coder = coder
        self.outer_loops = outer_loops
        self.inner_loops = inner_loops

        if execution is not None:
            self.execution = execution
            self.runner = runner or execution.runner
        else:
            if leakage is None or usage is None or debugger is None or runner is None:
                raise ValueError(
                    "Either execution or (leakage, usage, debugger, runner) must be provided."
                )
            self.execution = ExecutionGuardrailPipeline(
                config=ExecutionConfig(
                    timeout_seconds=runner.timeout_seconds,
                    max_debug_rounds=debugger.max_debug_rounds,
                    sandbox_base_dir=str(runner.runs_dir),
                ),
                leakage=leakage,
                usage=usage,
                runner=runner,
                debugger=debugger,
            )
            self.runner = runner
        self.leakage = self.execution.leakage
        self.usage = self.execution.usage
        self.debugger = self.execution.debugger

    def refine(
        self,
        spec: TaskSpecification,
        initial_code: str,
        initial_score: float | None,
        run_id: str = "refine",
    ) -> RefinementResult:
        """Run the nested refinement loops over the initial solution.

        Args:
            spec: The task specification.
            initial_code: The initial solution script ($s_0$).
            initial_score: The initial validation score ($h(s_0)$).
            run_id: Identifier of the current run.

        Returns:
            A ``RefinementResult`` with the best solution and lineage.
        """
        logger = CentralIterationLogger.for_run(self.runner.runs_dir, run_id)
        branch_index = branch_index_from_run_id(run_id)

        direction = spec.metric_direction
        current_code = initial_code
        current_score = initial_score
        final_code = initial_code
        final_score = initial_score
        lineage = [
            PipelineArtifact(
                version=0,
                full_code=initial_code,
                validation_score=initial_score,
                parent_version=None,
                applied_diff=None,
                iteration_stage="init",
            )
        ]
        ablation_history: list[str] = []
        refined_blocks: list[str] = []

        with logfire.span("refinement.run", run_id=run_id):
            for t in range(self.outer_loops):
                with logfire.span("refinement.outer", t=t):
                    announce(
                        f"[Outer {t + 1}/{self.outer_loops}] Running ablation study "
                        f"across components..."
                    )
                    try:
                        summary = self._outer_step(spec, current_code, ablation_history, run_id, t)
                    except Exception as exc:
                        logfire.warn("refinement.outer.failed", t=t, error=str(exc))
                        continue
                    ablation_history.append(summary)
                    logger.append(
                        IterationLogEntry(
                            iteration_id=f"t{t}_ablation",
                            stage="REFINEMENT",
                            hypothesis=summary[:1000],
                            code_diff="",
                            metrics={},
                            validation_score=None,
                            delta_from_baseline=None,
                            error_recovery_events=[],
                            success=True,
                            target_component="ABLATION_STUDY",
                            branch_index=branch_index,
                            duration_seconds=None,
                        )
                    )

                    try:
                        block, initial_plan = self.extractor.extract(
                            solution=current_code,
                            ablation_summary=summary,
                            previous_blocks=refined_blocks,
                        )
                    except Exception as exc:
                        logfire.warn("refinement.extract.failed", t=t, error=str(exc))
                        continue
                    announce(
                        f"[Outer {t + 1}/{self.outer_loops}] Extracted high-impact block: "
                        f"'{block.category.value}'"
                    )

                    best_inner_code = current_code
                    best_inner_score = current_score
                    attempts: list[tuple[str, float | None]] = []

                    for k in range(self.inner_loops):
                        if k == 0:
                            plan = initial_plan
                        else:
                            try:
                                plan = self.planner.next_plan(block, attempts, iteration_index=k)
                            except Exception as exc:
                                logfire.warn("refinement.planner.failed", t=t, k=k, error=str(exc))
                                break
                        with logfire.span("refinement.inner", t=t, k=k):
                            record, candidate_code = self._inner_step(
                                spec=spec,
                                block=block,
                                plan=plan,
                                base_code=current_code,
                                run_id=run_id,
                                t=t,
                                k=k,
                                initial_score=initial_score,
                                direction=direction,
                                logger=logger,
                                branch_index=branch_index,
                            )
                        attempts.append((plan.natural_language_plan, record.validation_score))
                        announce(
                            f"[Inner {t + 1}.{k + 1}/{self.inner_loops}] Plan: "
                            f"'{plan.natural_language_plan}' -> "
                            f"Score: {format_score(record.validation_score)} "
                            f"(Δ {format_delta(record.delta_from_baseline)})"
                        )

                        if (
                            record.success
                            and record.validation_score is not None
                            and candidate_code is not None
                        ):
                            if self._accepts(record.validation_score, best_inner_score, direction):
                                best_inner_score = record.validation_score
                                best_inner_code = candidate_code
                            if self._accepts(record.validation_score, final_score, direction):
                                final_score = record.validation_score
                                final_code = candidate_code
                                lineage.append(
                                    PipelineArtifact(
                                        version=len(lineage),
                                        full_code=final_code,
                                        validation_score=final_score,
                                        parent_version=len(lineage) - 1,
                                        applied_diff=record.code_diff or None,
                                        iteration_stage=f"outer{t}_inner{k}",
                                    )
                                )

                    if self._accepts(best_inner_score, current_score, direction):
                        current_code = best_inner_code
                        current_score = best_inner_score
                    refined_blocks.append(block.raw_code)

        return RefinementResult(
            final_code=final_code,
            final_score=final_score,
            lineage=lineage,
            logs_path=str(logger.logs_path),
            outer_iterations=self.outer_loops,
            inner_iterations=self.inner_loops,
        )

    def _outer_step(
        self,
        spec: TaskSpecification,
        current_code: str,
        ablation_history: list[str],
        run_id: str,
        iteration_index: int,
    ) -> str:
        """Run ablate -> summarize for one outer iteration.

        Args:
            spec: The task specification.
            current_code: The current solution script.
            ablation_history: Previous ablation summaries.
            run_id: Identifier of the current run.
            iteration_index: Zero-based outer-loop index, used to scope
                the ablation sandbox.

        Returns:
            The summarized ablation output for the extractor.
        """
        ablation_code = self.ablation.generate_ablation(current_code, ablation_history)
        report = self.summarizer.summarize(
            ablation_code,
            run_id=run_id,
            dataset_dir=spec.dataset_dir,
            dataset_files=spec.dataset_files,
            iteration_index=iteration_index,
        )
        return report.raw_log_summary or report.model_dump_json()

    def _inner_step(
        self,
        spec: TaskSpecification,
        block: TargetCodeBlock,
        plan: RefinementPlan,
        base_code: str,
        run_id: str,
        t: int,
        k: int,
        initial_score: float | None,
        direction: MetricDirection,
        logger: CentralIterationLogger,
        branch_index: int | None,
    ) -> tuple[IterationLogEntry, str | None]:
        """Run code -> patch -> guardrails -> evaluate for one attempt.

        Args:
            spec: The task specification.
            block: The target code block.
            plan: The refinement plan to implement.
            base_code: The solution the block is patched into.
            run_id: Identifier of the current run.
            t: Outer iteration index.
            k: Inner iteration index.
            initial_score: Score of the input solution ($h(s_0)$).
            direction: Metric direction for delta computation.
            logger: Central iteration logger to append the record to.
            branch_index: Parallel branch index, if any.

        Returns:
            The log entry and the candidate code (``None`` on failure).
        """
        errors: list[str] = []
        score: float | None = None
        success = False
        code_diff = ""
        duration_seconds: float | None = None
        candidate_code: str | None = None

        try:
            refined = self.coder.refine(block, plan)
            patched = patch_script(base_code, block.raw_code, refined)
            candidate_code = patched

            result = self.execution.run(
                patched,
                spec,
                run_id=run_id,
                candidate_id=f"refine_t{t}_k{k}",
            )
            guarded = self.execution.last_guarded_code or patched
            candidate_code = guarded
            score = result.validation_score
            success = score is not None
            duration_seconds = result.duration_seconds
            code_diff = compute_code_diff(base_code, guarded)
            if not success and result.stderr:
                errors.append(result.stderr[-500:])
            if guarded != patched:
                errors.append("guardrails modified the candidate")
        except ValueError as exc:
            errors.append(str(exc))
        except Exception as exc:
            errors.append(f"inner step failed: {exc}")

        record = IterationLogEntry(
            iteration_id=f"t{t}_k{k}",
            stage="REFINEMENT",
            hypothesis=plan.natural_language_plan,
            code_diff=code_diff,
            metrics=({"primary": score} if score is not None else {}),
            validation_score=score,
            delta_from_baseline=(
                direction.delta(score, initial_score)
                if score is not None and initial_score is not None
                else None
            ),
            error_recovery_events=errors,
            success=success,
            target_component=block.category.value,
            branch_index=branch_index,
            duration_seconds=duration_seconds,
        )
        logger.append(record)
        return record, candidate_code

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
