"""Sequential model merging agent ($A_merger$) implementing Algorithm 1.

Greedily blends the ranked candidate solutions into a consolidated
initial solution: it starts from the best candidate and merges each
subsequent candidate via the LLM (Figure 11 prompt), accepting a merge
only when the merged validation score is at least as good as the current
best, and aborting the loop on the first regression or failure.
"""

from __future__ import annotations

import logfire
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent

from problem_2_v2.contracts.code_utils import (
    compute_code_diff,
    extract_python_code,
    validate_python_syntax,
)
from problem_2_v2.contracts.enums import MetricDirection
from problem_2_v2.contracts.task import ExecutionResult, PipelineArtifact, TaskSpecification
from problem_2_v2.initialization.evaluator import CandidateEvaluation
from problem_2_v2.runner.debugger import DebuggerAgent

_MERGER_INSTRUCTIONS = (
    "You are a Kaggle grandmaster attending a competition.\n"
    "You will be given a base solution and an additional reference "
    "solution, and you need to implement your Python solution by "
    "integrating the reference solution into the base solution.\n"
    "# Your task\n"
    "- You have to integrate the reference solution to the base solution.\n"
    "- Your code base should be the base solution.\n"
    "- Try to train an additional model of the reference solution.\n"
    "- When integrating, try to keep code with similar functionality in "
    "the same place (e.g., all preprocessing should be done and then all "
    "training).\n"
    "- When integrating, ensemble the models.\n"
    "- The solution design should be relatively simple.\n"
    "- The code should implement the proposed solution and print the value "
    "of the evaluation metric computed on a hold-out validation set.\n"
    "- Only use the provided train data in the ./input directory.\n"
    "- If there are more than 30,000 training samples, you must subsample "
    "to 30,000 for a faster run.\n"
    "# Required\n"
    "- There should be no additional headings or text in your response.\n"
    "- Print out or return a final performance metric with the exact words: "
    "'Final Validation Performance: {final_validation_score}'.\n"
    "- The code should be a single-file Python program that is "
    "self-contained and can be executed as-is.\n"
    "- Your response should only contain a single code block.\n"
    "- Do not use exit() function in the Python code.\n"
    "- Do not use try: and except: or if else to ignore unintended behavior."
)


class MergeStep(BaseModel):
    """A single greedy merge attempt.

    Attributes:
        rank: Rank (k) of the candidate being merged.
        candidate_name: Name of the candidate being merged in.
        merged_code: The code produced by the merging agent.
        result: Execution result of the merged code.
        accepted: Whether the merge was accepted.
        reason: ``accepted``, ``rejected_score``, or ``rejected_error``.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    rank: int = Field(description="Rank of the merged-in candidate.")
    candidate_name: str = Field(description="Name of the merged-in candidate.")
    merged_code: str = Field(description="Code produced by the merging agent.")
    result: ExecutionResult | None = Field(
        default=None, description="Merged code execution result."
    )
    accepted: bool = Field(description="Whether the merge was accepted.")
    reason: str = Field(description="Accept/reject reason.")


class MergeOutcome(BaseModel):
    """Final outcome of the greedy sequential merging (Algorithm 1).

    Attributes:
        final_code: The consolidated initial solution code.
        final_score: Validation score of the final solution.
        lineage: Versioned artifact lineage of accepted merges.
        steps: All merge attempts in order.
        merged_count: Number of accepted merges.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    final_code: str = Field(description="Consolidated initial solution code.")
    final_score: float | None = Field(default=None, description="Final validation score.")
    lineage: list[PipelineArtifact] = Field(default_factory=list, description="Artifact lineage.")
    steps: list[MergeStep] = Field(default_factory=list, description="Merge attempts.")
    merged_count: int = Field(default=0, description="Accepted merge count.")


class ModelMergerAgent:
    """Greedily merges ranked candidate solutions into the initial $s_0$.

    Attributes:
        debugger: Debugger agent used to repair failing merged scripts.
        agent: Pydantic AI agent producing the merged Python script.
    """

    def __init__(
        self,
        debugger: DebuggerAgent,
        model: str = "openai:gpt-4o",
    ) -> None:
        """Create a model merger agent.

        Args:
            debugger: Debugger agent for merged-script repair.
            model: Pydantic AI model string.
        """
        self.debugger = debugger
        self.agent = Agent(
            model,
            name="model_merger_agent",
            output_type=str,
            instructions=_MERGER_INSTRUCTIONS,
            defer_model_check=True,
        )

    def merge(
        self,
        spec: TaskSpecification,
        ranked: list[CandidateEvaluation],
        run_id: str,
    ) -> MergeOutcome:
        """Run the greedy sequential merging procedure.

        Args:
            spec: The task specification (used for metric direction and
                dataset mapping).
            ranked: Candidate evaluations sorted best-first.
            run_id: Identifier of the current run.

        Returns:
            A ``MergeOutcome`` with the consolidated solution, its score,
            and the full merge history.
        """
        if not ranked:
            return MergeOutcome(final_code="", final_score=None)

        best: CandidateEvaluation = ranked[0]
        current_code = best.code
        best_score = best.score
        lineage: list[PipelineArtifact] = []
        steps: list[MergeStep] = []
        merged_count = 0

        if best.score is not None:
            lineage.append(
                PipelineArtifact(
                    version=0,
                    full_code=best.code,
                    validation_score=best.score,
                    parent_version=None,
                    applied_diff=None,
                    iteration_stage="init",
                )
            )

        for k in range(1, len(ranked)):
            candidate = ranked[k]
            with logfire.span(
                "merger.attempt",
                rank=k,
                candidate=candidate.model_name,
                current_best=best_score,
            ):
                prompt = (
                    f"# Base solution\n{current_code}\n"
                    f"# Reference solution\n{candidate.code}\n"
                    f"# Your task\nIntegrate the reference solution into "
                    f"the base solution and ensemble the models."
                )
                response = self.agent.run_sync(prompt)
                merged_code = extract_python_code(response.output)

                valid, error = validate_python_syntax(merged_code)
                if not valid or not merged_code:
                    steps.append(
                        MergeStep(
                            rank=k,
                            candidate_name=candidate.model_name,
                            merged_code=merged_code,
                            result=None,
                            accepted=False,
                            reason="rejected_error",
                        )
                    )
                    break

                outcome = self.debugger.debug(
                    merged_code,
                    run_id=run_id,
                    candidate_id=f"merge_{k}",
                    dataset_dir=spec.dataset_dir,
                    dataset_files=spec.dataset_files,
                )
                merged_score = outcome.result.validation_score

                accepted = self._accepts(spec.metric_direction, merged_score, best_score)
                if accepted:
                    merged_count += 1
                    best_score = merged_score
                    previous_code = current_code
                    current_code = outcome.code
                    lineage.append(
                        PipelineArtifact(
                            version=len(lineage),
                            full_code=outcome.code,
                            validation_score=merged_score,
                            parent_version=lineage[-1].version if lineage else None,
                            applied_diff=compute_code_diff(previous_code, outcome.code),
                            iteration_stage="merge",
                        )
                    )
                    steps.append(
                        MergeStep(
                            rank=k,
                            candidate_name=candidate.model_name,
                            merged_code=outcome.code,
                            result=outcome.result,
                            accepted=True,
                            reason="accepted",
                        )
                    )
                else:
                    steps.append(
                        MergeStep(
                            rank=k,
                            candidate_name=candidate.model_name,
                            merged_code=outcome.code,
                            result=outcome.result,
                            accepted=False,
                            reason="rejected_error" if merged_score is None else "rejected_score",
                        )
                    )
                    break

        return MergeOutcome(
            final_code=current_code,
            final_score=best_score,
            lineage=lineage,
            steps=steps,
            merged_count=merged_count,
        )

    @staticmethod
    def _accepts(
        direction: MetricDirection,
        candidate_score: float | None,
        best_score: float | None,
    ) -> bool:
        """Return whether a merged score is at least as good as the best."""
        if candidate_score is None:
            return False
        if best_score is None:
            return True
        return candidate_score == best_score or direction.is_better(candidate_score, best_score)
