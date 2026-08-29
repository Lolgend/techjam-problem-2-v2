"""Master orchestrator coordinating the full MLE-STAR pipeline.

Wires Task Ingestion -> Parallel Branches -> Adaptive Ensembling -> Final
Artifact Production -> Baseline Comparison into a single ``run`` /
``run_async`` entry point with zero manual intervention.
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

import logfire
from pydantic import BaseModel, ConfigDict, Field

from problem_2_v2.config import MLEStarConfig
from problem_2_v2.console import announce
from problem_2_v2.contracts.task import PipelineArtifact, TaskSpecification
from problem_2_v2.ensembling.ensembler import EnsemblerAgent
from problem_2_v2.ensembling.parallel import BranchBuilder, ParallelSolutionGenerator
from problem_2_v2.ensembling.pipeline import EnsemblePipeline, EnsembleResult
from problem_2_v2.ensembling.planner import EnsemblePlannerAgent
from problem_2_v2.execution.finalizer import FinalArtifact, FinalArtifactProducer
from problem_2_v2.execution.pipeline import ExecutionConfig, ExecutionGuardrailPipeline
from problem_2_v2.guardrails.leakage import DataLeakageCheckerAgent
from problem_2_v2.guardrails.usage import DataUsageCheckerAgent
from problem_2_v2.ingestion.extractor import TaskExtractor
from problem_2_v2.initialization.evaluator import CandidateEvaluatorAgent
from problem_2_v2.initialization.merger import ModelMergerAgent
from problem_2_v2.initialization.pipeline import InitializationPipeline
from problem_2_v2.refinement.ablation import AblationAgent, AblationSummarizerAgent
from problem_2_v2.refinement.coder import CoderAgent
from problem_2_v2.refinement.extractor import CodeBlockExtractorAgent
from problem_2_v2.refinement.pipeline import RefinementPipeline
from problem_2_v2.refinement.planner import RefinementPlannerAgent
from problem_2_v2.runner.debugger import DebuggerAgent
from problem_2_v2.runner.sandbox import SubprocessRunner
from problem_2_v2.search.providers import (
    DuckDuckGoSearchProvider,
    GoogleSearchProvider,
    MockSearchProvider,
    SearchProvider,
    TavilySearchProvider,
)
from problem_2_v2.search.retriever import RetrieverAgent


def configure_event_loop_policy() -> None:
    """Use the selector event loop policy on Windows.

    The default Proactor loop on Windows can raise ``OSError`` WinError
    10038 during socket teardown of ``asyncio.to_thread`` worker sockets in
    concurrent parallel branches. The selector policy avoids the Proactor
    ``_ProactorBasePipeTransport`` teardown path. It is a no-op on other
    platforms.
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class MLEStarResult(BaseModel):
    """Structured outcome of a full MLE-STAR run.

    Attributes:
        task_spec: The extracted task specification.
        branch_artifacts: Candidate artifacts from the parallel branches.
        ensemble_result: Outcome of the adaptive ensembling stage, if any.
        final_artifact: Production artifact, if produced.
        baseline_score: Official baseline score from the task spec.
        final_score: Validation score of the final production artifact.
        score_delta: ``final_score - baseline_score``.
        duration_seconds: Wall-clock duration of the run.
        success: Whether the run produced a final score.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_spec: TaskSpecification = Field(description="Extracted task specification.")
    branch_artifacts: list[PipelineArtifact] = Field(
        default_factory=list,
        description="Branch candidate artifacts.",
    )
    ensemble_result: EnsembleResult | None = Field(default=None, description="Ensembling outcome.")
    final_artifact: FinalArtifact | None = Field(default=None, description="Production artifact.")
    baseline_score: float = Field(description="Official baseline score.")
    final_score: float | None = Field(default=None, description="Final validation score.")
    score_delta: float | None = Field(default=None, description="Delta over the baseline.")
    duration_seconds: float = Field(description="Wall-clock run duration.")
    success: bool = Field(description="Whether the run produced a final score.")


class MLEStarPipeline:
    """Top-level coordinator of the complete MLE-STAR workflow.

    Attributes:
        config: Master hyperparameter configuration.
        runner: Shared sandbox runner.
        execution: Unified execution guardrail pipeline.
        ensembler: Code ensembler agent (execution-wired).
        ensemble_pipeline: Iterative ensemble pipeline.
        finalizer: Final artifact producer.
        parallel: Parallel branch solution generator.
    """

    def __init__(
        self,
        config: MLEStarConfig | None = None,
        *,
        branch_builder: BranchBuilder | None = None,
        search_provider: SearchProvider | None = None,
    ) -> None:
        """Create the master pipeline.

        Args:
            config: Master configuration (defaults to ``MLEStarConfig()``).
            branch_builder: Optional seed-aware factory for (initialization,
                refinement) pipeline pairs; defaults to the internal builder.
            search_provider: Optional pre-built search provider; otherwise one
                is constructed from ``config.search_provider``.
        """
        self.config = config or MLEStarConfig()
        self._provider = search_provider or self._build_provider()

        self.runner = SubprocessRunner(
            runs_dir=self.config.runs_dir,
            timeout_seconds=self.config.timeout_seconds,
        )
        self.execution = ExecutionGuardrailPipeline(
            config=ExecutionConfig(
                timeout_seconds=self.config.timeout_seconds,
                max_debug_rounds=self.config.max_debug_rounds,
                sandbox_base_dir=self.config.runs_dir,
                production_timeout_seconds=self.config.production_timeout_seconds,
            ),
            model=self.config.model,
        )
        self.debugger = self.execution.debugger
        self.finalizer = FinalArtifactProducer(
            debugger=self.debugger,
            model=self.config.model,
            config=ExecutionConfig(
                timeout_seconds=self.config.timeout_seconds,
                max_debug_rounds=self.config.max_debug_rounds,
                sandbox_base_dir=self.config.runs_dir,
                production_timeout_seconds=self.config.production_timeout_seconds,
            ),
        )
        self.ensembler = EnsemblerAgent(
            debugger=self.debugger,
            model=self.config.model,
            execution=self.execution,
        )
        self.ensemble_pipeline = EnsemblePipeline(
            planner=EnsemblePlannerAgent(model=self.config.model),
            ensembler=self.ensembler,
            runner=self.runner,
            rounds=self.config.ensemble_rounds,
            execution=self.execution,
        )
        self.parallel = ParallelSolutionGenerator(
            branch_builder=branch_builder or self._build_branch,
            num_branches=self.config.num_branches,
        )

    def validate(self, task_md_path: str, dataset_dir: str) -> TaskSpecification:
        """Validate run inputs without executing LLM code generation.

        Args:
            task_md_path: Path to the problem markdown file.
            dataset_dir: Path to the dataset directory.

        Returns:
            The parsed ``TaskSpecification`` (dry-run).

        Raises:
            FileNotFoundError: When the task markdown file is missing.
            NotADirectoryError: When the dataset directory is missing.
        """
        _, spec = self._ingest(task_md_path, dataset_dir)
        return spec

    def run(
        self,
        task_md_path: str,
        dataset_dir: str,
        run_id: str | None = None,
    ) -> MLEStarResult:
        """Run the full pipeline synchronously (blocking).

        Must be called from a thread without a running event loop; use
        ``run_async`` from within an active event loop.

        Args:
            task_md_path: Path to the problem markdown file.
            dataset_dir: Path to the dataset directory.
            run_id: Optional run identifier; auto-generated when omitted.

        Returns:
            An ``MLEStarResult`` describing the full run outcome.
        """
        configure_event_loop_policy()
        return asyncio.run(self.run_async(task_md_path, dataset_dir, run_id=run_id))

    async def run_async(
        self,
        task_md_path: str,
        dataset_dir: str,
        run_id: str | None = None,
    ) -> MLEStarResult:
        """Run the 5-stage MLE-STAR workflow asynchronously.

        Args:
            task_md_path: Path to the problem markdown file.
            dataset_dir: Path to the dataset directory.
            run_id: Optional run identifier; auto-generated when omitted.

        Returns:
            An ``MLEStarResult`` describing the full run outcome.
        """
        started = time.monotonic()
        run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with logfire.span("mlestar.run", run_id=run_id):
            md_text, spec = self._ingest(task_md_path, dataset_dir)
            baseline = spec.baseline_score

            seeds = (
                self.config.seeds
                if self.config.seeds is not None
                else list(range(self.config.num_branches))
            )
            announce(f"[Stage 1/4] Launching {len(seeds)} Parallel Seed Branches...")
            with logfire.span("mlestar.parallel_branches", run_id=run_id):
                artifacts = await self.parallel.generate(
                    md_text, dataset_dir, run_id=run_id, seeds=seeds
                )
            announce(
                f"[Stage 2/4] Aggregating Candidate Artifacts ({len(artifacts)} successful)..."
            )

            ensemble_result: EnsembleResult | None = None
            if artifacts:
                announce(
                    f"[Stage 3/4] Adaptive Ensembling ({self.config.ensemble_rounds} rounds)..."
                )
                with logfire.span("mlestar.ensembling", run_id=run_id):
                    ensemble_result = await asyncio.to_thread(
                        self.ensemble_pipeline.run, spec, artifacts, run_id
                    )
            else:
                import sys

                print(
                    "[MLE-STAR] Warning: No candidate solutions were produced "
                    "across parallel branches.",
                    file=sys.stderr,
                )

            final_artifact: FinalArtifact | None = None
            best_code = (
                ensemble_result.best_code
                if ensemble_result is not None
                else (artifacts[0].full_code if artifacts else None)
            )
            if best_code:
                announce(
                    "[Stage 4/4] Production Finalization "
                    "(Full Dataset Training & Model Serialization)..."
                )
                with logfire.span("mlestar.finalization", run_id=run_id):
                    final_artifact = await asyncio.to_thread(
                        self.finalizer.produce, best_code, spec, f"{run_id}/final"
                    )

            final_score = final_artifact.validation_score if final_artifact is not None else None
            score_delta = final_score - baseline if final_score is not None else None

        return MLEStarResult(
            task_spec=spec,
            branch_artifacts=artifacts,
            ensemble_result=ensemble_result,
            final_artifact=final_artifact,
            baseline_score=baseline,
            final_score=final_score,
            score_delta=score_delta,
            duration_seconds=time.monotonic() - started,
            success=final_score is not None,
        )

    def _ingest(self, task_md_path: str, dataset_dir: str) -> tuple[str, TaskSpecification]:
        """Validate paths, read the markdown, and parse the task spec once."""
        task_path = Path(task_md_path)
        if not task_path.is_file():
            raise FileNotFoundError(f"Task markdown file not found: {task_md_path}")
        data_path = Path(dataset_dir)
        if not data_path.is_dir():
            raise NotADirectoryError(f"Dataset directory not found: {dataset_dir}")
        md_text = task_path.read_text(encoding="utf-8")
        return md_text, TaskSpecification.from_markdown(md_text, dataset_dir=dataset_dir)

    def _build_provider(self) -> SearchProvider:
        """Build the configured search provider backend."""
        name = self.config.search_provider
        if name == "mock":
            return MockSearchProvider()
        if name == "tavily":
            return TavilySearchProvider()
        if name == "google":
            return GoogleSearchProvider()
        return DuckDuckGoSearchProvider()

    def _build_branch(self, seed: int) -> tuple[InitializationPipeline, RefinementPipeline]:
        """Build a fresh (initialization, refinement) pair for a seed.

        Each branch owns its runner and debugger so sandbox namespaces stay
        isolated across the concurrent branches.
        """
        runner = SubprocessRunner(
            runs_dir=self.config.runs_dir,
            timeout_seconds=self.config.timeout_seconds,
        )
        debugger = DebuggerAgent(
            runner=runner,
            model=self.config.model,
            max_debug_rounds=self.config.max_debug_rounds,
        )
        init = InitializationPipeline(
            extractor=TaskExtractor(model=self.config.model),
            retriever=RetrieverAgent(
                provider=self._provider,
                model=self.config.model,
                num_candidates=self.config.num_candidates,
            ),
            evaluator=CandidateEvaluatorAgent(debugger=debugger, model=self.config.model),
            merger=ModelMergerAgent(debugger=debugger, model=self.config.model),
        )
        refine = RefinementPipeline(
            ablation=AblationAgent(model=self.config.model),
            summarizer=AblationSummarizerAgent(runner=runner, model=self.config.model),
            extractor=CodeBlockExtractorAgent(model=self.config.model),
            planner=RefinementPlannerAgent(model=self.config.model),
            coder=CoderAgent(model=self.config.model),
            leakage=DataLeakageCheckerAgent(model=self.config.model),
            usage=DataUsageCheckerAgent(model=self.config.model),
            debugger=debugger,
            runner=runner,
            outer_loops=self.config.outer_loops,
            inner_loops=self.config.inner_loops,
        )
        return init, refine
