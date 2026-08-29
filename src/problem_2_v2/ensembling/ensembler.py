"""Code ensembler agent ($A_ensembler$).

Synthesizes a single-file, self-contained Python program that combines the
candidate solutions according to the active ensemble plan (Figure 18
prompt), executes it in the sandbox with the debugger fallback, and
records the validation score and submission file location.
"""

from __future__ import annotations

from pathlib import Path

import logfire
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent

from problem_2_v2.contracts.code_utils import extract_python_code, validate_python_syntax
from problem_2_v2.contracts.guardrails import EnsembleStrategy
from problem_2_v2.contracts.task import ExecutionResult, PipelineArtifact, TaskSpecification
from problem_2_v2.runner.debugger import DebuggerAgent

_ENSEMBLER_INSTRUCTIONS = (
    "You are a Kaggle grandmaster attending a competition. In order to win "
    "this competition, you need to ensemble the provided Python Solutions "
    "for better performance based on the ensemble plan.\n"
    "# Your task\n"
    "- Implement the ensemble plan with the provided solutions.\n"
    "- Unless mentioned in the ensemble plan, do not modify the original "
    "Python Solutions too much.\n"
    "- All the provided data (except previous submissions; do not load "
    "submissions) is already prepared and available in the ./input "
    "directory. There is no need to unzip any files.\n"
    "- The code should implement the proposed solution and print the value "
    "of the evaluation metric computed on a hold-out validation set.\n"
    "# Response format\n"
    "- Your response should be a single markdown code block (wrapped in "
    "```) which is the ensemble of the provided Python Solutions.\n"
    "- There should be no additional headings or text in your response.\n"
    "- Do not subsample or introduce dummy variables. Provide a full new "
    "Python Solution using the provided solutions.\n"
    "- Do not forget the ./final/submission.csv file.\n"
    "- Print out or return a final performance metric with the exact words: "
    "'Final Validation Performance: {final_validation_score}'.\n"
    "- The code should be a single-file Python program that is "
    "self-contained and can be executed as-is."
)


class EnsembleRun(BaseModel):
    """Outcome of implementing one ensemble plan.

    Attributes:
        round_index: Ensemble round index (r).
        code: Final (possibly debugged) ensemble script.
        result: Final execution result, if executed.
        score: Validation score, or ``None`` on failure.
        debug_rounds: Debugger repair rounds used.
        submission_path: Path to ``./final/submission.csv``, if created.
        success: Whether the script ran with a validation score.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    round_index: int = Field(description="Ensemble round index.")
    code: str = Field(description="Final ensemble script.")
    result: ExecutionResult | None = Field(default=None, description="Execution result.")
    score: float | None = Field(default=None, description="Validation score.")
    debug_rounds: int = Field(default=0, description="Debugger rounds used.")
    submission_path: str | None = Field(default=None, description="Submission file path.")
    success: bool = Field(description="Whether the run produced a score.")


class EnsemblerAgent:
    """Implements ensemble plans as executable single-file scripts.

    Attributes:
        debugger: Debugger agent used to repair failing ensemble scripts.
        agent: Pydantic AI agent producing the unified ensemble script.
    """

    def __init__(self, debugger: DebuggerAgent, model: str = "openai:gpt-4o") -> None:
        """Create an ensembler.

        Args:
            debugger: Debugger agent for execution repair.
            model: Pydantic AI model string.
        """
        self.debugger = debugger
        self.agent = Agent(
            model,
            name="ensembler_agent",
            output_type=str,
            instructions=_ENSEMBLER_INSTRUCTIONS,
            defer_model_check=True,
        )

    def ensemble(
        self,
        spec: TaskSpecification,
        solutions: list[PipelineArtifact],
        strategy: EnsembleStrategy,
        run_id: str,
        round_index: int,
    ) -> EnsembleRun:
        """Implement an ensemble plan and evaluate the merged script.

        Args:
            spec: The task specification.
            solutions: The candidate solution artifacts to combine.
            strategy: The active ensemble strategy.
            run_id: Identifier of the current run.
            round_index: Ensemble round index (r).

        Returns:
            An ``EnsembleRun`` with the final script, score, and
            submission file location.
        """
        prompt = self.build_prompt(solutions, strategy)
        with logfire.span("ensembler.generate", round=round_index, method=strategy.method.value):
            response = self.agent.run_sync(prompt)
        code = extract_python_code(response.output)

        valid, error = validate_python_syntax(code)
        if not valid:
            logfire.warn(
                "ensembler.invalid_code; handing to debugger", round=round_index, error=error
            )

        if not code:
            return EnsembleRun(
                round_index=round_index,
                code="",
                result=ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="Ensembler produced no code.",
                    returncode=-1,
                    duration_seconds=0.0,
                ),
                score=None,
                debug_rounds=0,
                submission_path=None,
                success=False,
            )

        candidate_id = f"ens_r{round_index}"
        outcome = self.debugger.debug(
            code,
            run_id=run_id,
            candidate_id=candidate_id,
            dataset_dir=spec.dataset_dir,
            dataset_files=spec.dataset_files,
        )
        score = outcome.result.validation_score
        submission = (
            Path(self.debugger.runner.runs_dir)
            / run_id
            / f"sandbox_{candidate_id}"
            / "final"
            / "submission.csv"
        )
        return EnsembleRun(
            round_index=round_index,
            code=outcome.code,
            result=outcome.result,
            score=score,
            debug_rounds=outcome.debug_rounds,
            submission_path=str(submission) if submission.exists() else None,
            success=score is not None,
        )

    @staticmethod
    def build_prompt(
        solutions: list[PipelineArtifact],
        strategy: EnsembleStrategy,
    ) -> str:
        """Build the Figure 18 ensemble implementation prompt.

        Args:
            solutions: The candidate solution artifacts.
            strategy: The active ensemble strategy.

        Returns:
            The full ensembling prompt.
        """
        solution_blocks = "\n".join(
            f"# {index + 1}th Python Solution\n{artifact.full_code}"
            for index, artifact in enumerate(solutions)
        )
        return (
            f"# Introduction\nYou are a Kaggle grandmaster attending a "
            f"competition.\n{solution_blocks}\n"
            f"# Ensemble Plan\n{strategy.natural_language_plan}\n"
            f"# Your task\nImplement the ensemble plan with the provided "
            f"solutions. Write the submission to ./final/submission.csv and "
            f"print the validation performance."
        )
