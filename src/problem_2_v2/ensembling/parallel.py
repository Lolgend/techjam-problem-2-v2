"""Parallel candidate solution generation across distinct seeds.

Runs the full Initialization + Refinement pipeline in $L$ concurrent
branches (via ``asyncio`` with thread offloading), each with its own
isolated sandbox namespace and random seed, producing $L$ validated
``PipelineArtifact`` instances.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import logfire

from problem_2_v2.contracts.task import PipelineArtifact
from problem_2_v2.initialization.pipeline import InitializationPipeline
from problem_2_v2.refinement.pipeline import RefinementPipeline, RefinementResult

BranchBuilder = Callable[[int], tuple[InitializationPipeline, RefinementPipeline]]


class ParallelSolutionGenerator:
    """Generates $L$ candidate solutions from seeded parallel branches.

    Attributes:
        branch_builder: Factory creating a fresh (initialization,
            refinement) pipeline pair for a given seed. Pipelines must be
            fully configured (models, runners) before being returned.
        num_branches: Number of branches to run when no explicit seeds are
            given.
    """

    def __init__(self, branch_builder: BranchBuilder, num_branches: int = 2) -> None:
        """Create a parallel solution generator.

        Args:
            branch_builder: Seed-aware factory for pipeline pairs.
            num_branches: Default number of branches (L).
        """
        self.branch_builder = branch_builder
        self.num_branches = num_branches

    async def generate(
        self,
        md_text: str,
        dataset_dir: str,
        run_id: str,
        seeds: list[int] | None = None,
    ) -> list[PipelineArtifact]:
        """Generate candidate solutions concurrently across seeds.

        Args:
            md_text: The raw markdown problem description.
            dataset_dir: Absolute path to the dataset directory.
            run_id: Identifier of the ensemble run; each branch is
                namespaced under ``run_id/branch_<i>``.
            seeds: Distinct random seeds per branch; defaults to
                ``[0, ..., L-1]``.

        Returns:
            The validated artifacts of successful branches (failed
            branches are skipped with a warning).
        """
        if seeds is None:
            seeds = list(range(self.num_branches))
        if len(set(seeds)) != len(seeds):
            raise ValueError("Seeds must be distinct per branch.")

        async def run_branch(seed: int, index: int) -> PipelineArtifact | None:
            with logfire.span("parallel.branch", index=index, seed=seed):
                try:
                    init_pipeline, refine_pipeline = self.branch_builder(seed)
                except Exception as exc:
                    logfire.warn(
                        "parallel.branch_setup_failed", index=index, seed=seed, error=str(exc)
                    )
                    return None
                branch_run_id = f"{run_id}/branch_{index}"
                try:
                    init_result = await asyncio.to_thread(
                        init_pipeline.run, md_text, dataset_dir, branch_run_id
                    )
                    refine_result = await asyncio.to_thread(
                        refine_pipeline.refine,
                        init_result.task,
                        init_result.best_code,
                        init_result.best_score,
                        branch_run_id,
                    )
                except Exception as exc:
                    logfire.warn("parallel.branch_failed", index=index, seed=seed, error=str(exc))
                    return None
                return self._to_artifact(refine_result, index, seed)

        results = await asyncio.gather(
            *(run_branch(seed, index) for index, seed in enumerate(seeds))
        )
        return [artifact for artifact in results if artifact is not None]

    @staticmethod
    def _to_artifact(refine_result: RefinementResult, index: int, seed: int) -> PipelineArtifact:
        """Convert a refinement result into a branch artifact.

        The branch identity (``branch_<i>_seed_<s>``) is preserved so it
        can serve as a unique candidate id in ensembling, while the
        lineage diff from the refinement tail is carried over.
        """
        lineage = refine_result.lineage
        parent = lineage[-1] if lineage else None
        return PipelineArtifact(
            version=(parent.version + 1) if parent else 0,
            full_code=refine_result.final_code,
            validation_score=refine_result.final_score,
            parent_version=parent.version if parent else None,
            applied_diff=parent.applied_diff if parent else None,
            iteration_stage=f"branch_{index}_seed_{seed}",
        )
