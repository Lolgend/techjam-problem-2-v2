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
from problem_2_v2.execution.pipeline import ExecutionGuardrailPipeline
from problem_2_v2.runner.debugger import DebuggerAgent

_ENSEMBLER_PROMPT_TEMPLATE = (
    "# Introduction\n"
    "- You are a Kaggle grandmaster attending a competition.\n"
    "- In order to win this competition, you need to ensemble {L} Python Solutions for better\n"
    "performance based on the ensemble plan.\n"
    "- We will now provide the Python Solutions and the ensemble plan.\n"
    "{solutions_block}\n"
    "# Ensemble Plan\n"
    "{plan}\n"
    "# Your task\n"
    "- Implement the ensemble plan with the provided solutions.\n"
    "- Unless mentioned in the ensemble plan, do not modify the original Python Solutions too\n"
    "much.\n"
    "- All the provided data (except previous submissions; do not load submissions) is already\n"
    "prepared and available in the `.\\input` directory. There is no need to unzip any files.\n"
    "- The code should implement the proposed solution and print the value of the evaluation\n"
    "metric computed on a hold-out validation set.\n"
    "# Response format required\n"
    "- Your response should be a single markdown code block (wrapped in ```) which is the\n"
    "ensemble of {L} Python Solutions.\n"
    "- There should be no additional headings or text in your response.\n"
    "- Do not subsample or introduce dummy variables. You have to provide full new Python\n"
    "Solution using the {L} provided solutions.\n"
    "- Do not forget the `./final/submission.csv` file.\n"
    "- Print out or return a final performance metric in your answer in a clear format with the "
    "exact words: 'Final Validation Performance: {{final_validation_score}}'.\n"
    "- The code should be a single-file Python program that can be executed as-is."
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
        execution: Optional unified execution guardrail pipeline; when
            provided, ensemble scripts are run through it instead of the
            debugger directly.
        agent: Pydantic AI agent producing the unified ensemble script.
    """

    def __init__(
        self,
        debugger: DebuggerAgent,
        model: str = "openai:gpt-4o",
        execution: ExecutionGuardrailPipeline | None = None,
    ) -> None:
        """Create an ensembler.

        Args:
            debugger: Debugger agent for execution repair.
            model: Pydantic AI model string.
            execution: Optional unified execution guardrail pipeline.
        """
        self.debugger = debugger
        self.execution = execution
        self.agent = Agent(
            model,
            name="ensembler_agent",
            output_type=str,
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
        if self.execution is not None:
            result = self.execution.run(code, spec, run_id=run_id, candidate_id=candidate_id)
            final_code = self.execution.last_executed_code or code
            debug_rounds = self.execution.last_debug_rounds
            runs_dir = self.execution.runner.runs_dir
        else:
            outcome = self.debugger.debug(
                code,
                run_id=run_id,
                candidate_id=candidate_id,
                dataset_dir=spec.dataset_dir,
                dataset_files=spec.dataset_files,
            )
            result = outcome.result
            final_code = outcome.code
            debug_rounds = outcome.debug_rounds
            runs_dir = self.debugger.runner.runs_dir
        score = result.validation_score
        submission = (
            Path(runs_dir) / run_id / f"sandbox_{candidate_id}" / "final" / "submission.csv"
        )
        return EnsembleRun(
            round_index=round_index,
            code=final_code,
            result=result,
            score=score,
            debug_rounds=debug_rounds,
            submission_path=str(submission) if submission.exists() else None,
            success=score is not None,
        )

    @staticmethod
    def build_prompt(
        solutions: list[PipelineArtifact],
        strategy: EnsembleStrategy,
    ) -> str:
        """Build the Figure 20 ensemble implementation prompt.

        Args:
            solutions: The candidate solution artifacts.
            strategy: The active ensemble strategy.

        Returns:
            The full ensembling prompt.
        """

        def _ordinal(n: int) -> str:
            suffix = (
                "th" if 11 <= (n % 100) <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
            )
            return f"{n}{suffix}"

        solution_parts = [
            f"# {_ordinal(i + 1)} Python Solution\n{artifact.full_code}"
            for i, artifact in enumerate(solutions)
        ]
        solutions_block = "\n".join(solution_parts)

        return _ENSEMBLER_PROMPT_TEMPLATE.format(
            L=len(solutions),
            solutions_block=solutions_block,
            plan=strategy.natural_language_plan,
        )

    _build_prompt = build_prompt
