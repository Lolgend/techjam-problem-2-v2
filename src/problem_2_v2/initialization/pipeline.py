"""Initialization pipeline coordinator.

Wires the task extractor, retriever, candidate evaluator, and model
merger into a single callable pipeline producing the consolidated initial
solution $s_0$ with a validated baseline score, following Algorithm 1 of
the MLE-STAR paper.
"""

from __future__ import annotations

import logfire
from pydantic import BaseModel, ConfigDict, Field

from problem_2_v2.contracts.search import RetrievedCandidates
from problem_2_v2.contracts.task import TaskSpecification
from problem_2_v2.ingestion.extractor import TaskExtractor
from problem_2_v2.initialization.evaluator import CandidateEvaluation, CandidateEvaluatorAgent
from problem_2_v2.initialization.merger import MergeOutcome, ModelMergerAgent
from problem_2_v2.search.retriever import RetrieverAgent


class InitializationResult(BaseModel):
    """Full outcome of an initialization run.

    Attributes:
        task: The extracted task specification.
        candidates: Retrieved candidate model cards.
        evaluations: Per-candidate evaluation results.
        outcome: Greedy merging outcome (final $s_0$ and lineage).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task: TaskSpecification = Field(description="Extracted task specification.")
    candidates: RetrievedCandidates = Field(description="Retrieved candidate cards.")
    evaluations: list[CandidateEvaluation] = Field(description="Per-candidate evaluations.")
    outcome: MergeOutcome = Field(description="Final merging outcome.")

    @property
    def best_code(self) -> str:
        """The final consolidated initial solution code."""
        return self.outcome.final_code

    @property
    def best_score(self) -> float | None:
        """The validated score of the initial solution."""
        return self.outcome.final_score


class InitializationPipeline:
    """Coordinates the full task ingestion -> merging initialization flow.

    Attributes:
        extractor: Task ingestion agent.
        retriever: Search-guided candidate retriever.
        evaluator: Candidate code generation and evaluation agent.
        merger: Greedy sequential model merger.
    """

    def __init__(
        self,
        extractor: TaskExtractor,
        retriever: RetrieverAgent,
        evaluator: CandidateEvaluatorAgent,
        merger: ModelMergerAgent,
    ) -> None:
        """Create the initialization pipeline.

        Args:
            extractor: Task extraction agent.
            retriever: Candidate retriever agent.
            evaluator: Candidate evaluation agent.
            merger: Sequential model merger agent.
        """
        self.extractor = extractor
        self.retriever = retriever
        self.evaluator = evaluator
        self.merger = merger

    def run(self, md_text: str, dataset_dir: str, run_id: str) -> InitializationResult:
        """Run the full initialization workflow.

        Args:
            md_text: The raw markdown problem description.
            dataset_dir: Absolute path to the dataset directory.
            run_id: Identifier of this run (used for sandbox paths).

        Returns:
            An ``InitializationResult`` with the task spec, retrieved
            candidates, evaluations, and the merged initial solution.
        """
        with logfire.span("initialization.run", run_id=run_id):
            with logfire.span("initialization.extract"):
                spec = self.extractor.extract(md_text, dataset_dir=dataset_dir)
            with logfire.span("initialization.retrieve"):
                candidates = self.retriever.retrieve(spec)
            with logfire.span(
                "initialization.evaluate",
                num_candidates=len(candidates.candidates),
            ):
                evaluations = self.evaluator.evaluate_all(
                    spec, candidates.candidates, run_id=run_id
                )
            ranked = self.evaluator.ranking(evaluations, spec.metric_direction)
            with logfire.span("initialization.merge", num_candidates=len(ranked)):
                outcome = self.merger.merge(spec, ranked, run_id=run_id)
        return InitializationResult(
            task=spec,
            candidates=candidates,
            evaluations=evaluations,
            outcome=outcome,
        )
