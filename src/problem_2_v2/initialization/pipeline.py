"""Initialization pipeline coordinator.

Wires the task extractor, retriever, candidate evaluator, and model
merger into a single callable pipeline producing the consolidated initial
solution $s_0$ with a validated baseline score, following Algorithm 1 of
the MLE-STAR paper.
"""

from __future__ import annotations

from pathlib import Path

import logfire
from pydantic import BaseModel, ConfigDict, Field

from problem_2_v2.console import announce, format_score
from problem_2_v2.contracts.search import ModelCard, RetrievedCandidates
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
        use_baseline: Whether to seed the official baseline starter script
            as the first candidate.
        baseline_path: Optional explicit path to the baseline starter script;
            otherwise standard workspace locations are searched.
    """

    def __init__(
        self,
        extractor: TaskExtractor,
        retriever: RetrieverAgent,
        evaluator: CandidateEvaluatorAgent,
        merger: ModelMergerAgent,
        use_baseline: bool = False,
        baseline_path: str | None = None,
    ) -> None:
        """Create the initialization pipeline.

        Args:
            extractor: Task extraction agent.
            retriever: Candidate retriever agent.
            evaluator: Candidate evaluation agent.
            merger: Sequential model merger agent.
            use_baseline: Seed the official baseline starter script as the
                first candidate card when one is found.
            baseline_path: Optional explicit baseline starter script path;
                otherwise ``src/baseline/baseline.py``, ``baseline.py``, and
                the dataset directory are searched in order.
        """
        self.extractor = extractor
        self.retriever = retriever
        self.evaluator = evaluator
        self.merger = merger
        self.use_baseline = use_baseline
        self.baseline_path = baseline_path

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
            announce(
                f"[Search] Retrieving candidates via {self.retriever.provider.provider_name}..."
            )
            with logfire.span("initialization.retrieve"):
                candidates = self.retriever.retrieve(spec)
            if self.use_baseline:
                baseline_code = self._load_baseline(spec)
                if baseline_code:
                    cards = [self._baseline_card(baseline_code)] + list(candidates.candidates)
                    candidates = RetrievedCandidates(
                        candidates=cards,
                        query_used=candidates.query_used,
                        total_found=len(cards),
                    )
                    announce("[Baseline] Official baseline starter script seeded as Candidate 1.")
            with logfire.span(
                "initialization.evaluate",
                num_candidates=len(candidates.candidates),
            ):
                evaluations = self.evaluator.evaluate_all(
                    spec, candidates.candidates, run_id=run_id
                )
            for index, evaluation in enumerate(evaluations):
                if evaluation.score is None:
                    reason = (
                        evaluation.result.stderr[-200:]
                        if evaluation.result is not None and evaluation.result.stderr
                        else "no validation score produced"
                    )
                    announce(
                        f"[Candidate {index + 1}/{len(evaluations)}] {evaluation.model_name} "
                        f"-> Failed: {reason.strip()}"
                    )
                else:
                    announce(
                        f"[Candidate {index + 1}/{len(evaluations)}] {evaluation.model_name} -> "
                        f"Validation Score: {format_score(evaluation.score)}"
                    )
            ranked = self.evaluator.ranking(evaluations, spec.metric_direction)
            with logfire.span("initialization.merge", num_candidates=len(ranked)):
                outcome = self.merger.merge(spec, ranked, run_id=run_id)
            announce(
                f"[Merge] Sequential merging completed. Initial s0 Score: "
                f"{format_score(outcome.final_score)}"
            )
        return InitializationResult(
            task=spec,
            candidates=candidates,
            evaluations=evaluations,
            outcome=outcome,
        )

    def _load_baseline(self, spec: TaskSpecification) -> str | None:
        """Return the official baseline starter script, or ``None``.

        Searches an explicit ``baseline_path``, then ``src/baseline/baseline.py``,
        ``baseline.py``, and ``<dataset_dir>/baseline.py`` in the workspace.
        """
        locations: list[str] = []
        if self.baseline_path is not None:
            locations.append(self.baseline_path)
        locations.extend(
            [
                "src/baseline/baseline.py",
                "baseline.py",
                str(Path(spec.dataset_dir) / "baseline.py"),
            ]
        )
        for location in locations:
            path = Path(location)
            if path.is_file():
                return path.read_text(encoding="utf-8", errors="replace")
        return None

    @staticmethod
    def _baseline_card(code: str) -> ModelCard:
        """Build an official-baseline model card from a starter script."""
        return ModelCard(
            model_name="Official Baseline",
            rationale="Official baseline starter script evaluated as a candidate.",
            example_code=code,
        )
