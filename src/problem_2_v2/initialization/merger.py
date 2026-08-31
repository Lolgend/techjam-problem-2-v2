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

_MERGER_PROMPT_TEMPLATE = (
    "# Introduction\n"
    "- You are a Kaggle grandmaster attending a competition.\n"
    "- We will now provide a base solution and an additional reference solution.\n"
    "- You need to implement your Python solution by integrating reference solution to the base\n"
    "solution.\n"
    "# Base solution\n"
    "{base_code}\n"
    "# Reference solution\n"
    "{reference_code}\n"
    "# Your task\n"
    "- Implement the solution in Python.\n"
    "- You have to integrate the reference solution to the base solution.\n"
    "- Your code base should be the base solution.\n"
    "- Try to train additional model of the reference solution.\n"
    "- When integrating, try to keep code with similar functionality in the same place (e.g.,\n"
    "all preprocessing should be done and then all training).\n"
    "- When integrating, ensemble the models.\n"
    "- The solution design should be relatively simple.\n"
    "- The code should implement the proposed solution and print the value of the evaluation\n"
    "metric computed on a hold-out validation set.\n"
    "- Only use the provided train data in the `./input` directory.\n"
    "- If there are more than 30,000 training samples, you must subsample to 30,000 for a faster\n"
    "run.\n"
    "# Required\n"
    "- There should be no additional headings or text in your response.\n"
    "- Print out or return a final performance metric in your answer in a clear format with the\n"
    "exact words: 'Final Validation Performance: {{final_validation_score}}'.\n"
    "- The code should be a single-file Python program that can be executed as-is.\n"
    "- Your response should only contain a single code block.\n"
    "- Do not use exit() function in the Python code.\n"
    "- Do not use try: and except: or if else to ignore unintended behavior\n"
    "### Evaluation Protocol & Metric Tooling (MANDATORY)\n"
    "The merged model must strictly evaluate validation performance using the official `evaluate.py` evaluation harness (used both in base and reference), it is model-agnostic — `evaluate(user_ids: Any, labels: Any, scores: Any) -> dict[str, Any]`, so any model can be scored with it:\n"
    "```python\n"
    "from evaluate import evaluate\n"
    "# user_ids: sequence of validation user IDs\n"
    "# labels: sequence of validation binary labels (0 or 1)\n"
    "# scores: continuous real-valued prediction scores from model\n"
    "val_res = evaluate(val_user_ids, val_labels, val_predictions)\n"
    "print(f'Final Validation Performance: {val_res['primary']:.6f}')\n"
    "```\n"
    "The metric `primary` is the arithmetic mean of Group AUC (GAUC) and normalized Discounted Cumulative Gain at rank 5 (nDCG@5):\n"
    "`primary = (GAUC + nDCG@5) / 2.0`\n"
    "- **GAUC:** Evaluated strictly on discriminative users where `0 < positive_count < impression_count`, weighted by positive impressions.\n"
    "- **nDCG@5:** Evaluated per user with gain `2^rel - 1`. All-negative users (27.1% of dataset) receive 0.0 and are included in the mean.\n"
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
            defer_model_check=True,
        )

    @staticmethod
    def build_prompt(base_code: str, reference_code: str) -> str:
        """Build the Figure 11 model merging prompt.

        Args:
            base_code: Python code of the current best base solution.
            reference_code: Python code of the reference candidate to merge.

        Returns:
            The formatted model merging prompt string.
        """
        return _MERGER_PROMPT_TEMPLATE.format(
            base_code=base_code,
            reference_code=reference_code,
        )

    _build_prompt = build_prompt

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

        best = next((evaluation for evaluation in ranked if evaluation.code.strip()), None)
        if best is None:
            return MergeOutcome(final_code="", final_score=None)
        current_code = best.code
        best_score = best.score
        lineage: list[PipelineArtifact] = []
        steps: list[MergeStep] = []
        merged_count = 0

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
                prompt = self.build_prompt(
                    base_code=current_code,
                    reference_code=candidate.code,
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
