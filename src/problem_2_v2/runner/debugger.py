"""Autonomous debugging agent ($A_debugger$).

When a generated script fails to execute (non-zero return code, timeout,
or a missing validation score line), the debugger feeds the code and its
traceback to the LLM (Figure 19 prompt) and re-runs the repaired script,
looping up to ``max_debug_rounds``.
"""

from __future__ import annotations

import logfire
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent

from problem_2_v2.contracts.task import ExecutionResult
from problem_2_v2.runner.sandbox import SubprocessRunner

_DEBUGGER_INSTRUCTIONS = (
    "You repair broken Python machine learning scripts.\n"
    "# Your task\n"
    "- Revise the code to fix the error.\n"
    "- Do not remove subsampling if exists.\n"
    "- Provide the improved, self-contained Python script again.\n"
    "- There should be no additional headings or text in your response.\n"
    '- All the provided input data is stored in "./input" directory.\n'
    "- Remember to print a line in the code with "
    "'Final Validation Performance: {final_validation_score}' so we can "
    "parse performance.\n"
    "- The code should be a single-file python program that is "
    "self-contained and can be executed as-is.\n"
    "- Your response should only contain a single code block.\n"
    "- Do not use exit() function in the refined Python code."
)


class DebugOutcome(BaseModel):
    """Result of a debugging session.

    Attributes:
        code: The final code after debugging (repaired or original).
        result: The final execution result.
        debug_rounds: Number of repair rounds actually performed.
        recovered: Whether the script eventually executed successfully
            with a validation score.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    code: str = Field(description="Final code after debugging.")
    result: ExecutionResult = Field(description="Final execution result.")
    debug_rounds: int = Field(description="Number of repair rounds performed.")
    recovered: bool = Field(description="Whether execution eventually succeeded.")


class DebuggerAgent:
    """Iteratively repairs failing scripts within a round budget.

    Attributes:
        runner: The sandbox runner used to execute scripts.
        agent: Pydantic AI agent producing repaired Python source.
        max_debug_rounds: Maximum repair attempts before giving up.
    """

    def __init__(
        self,
        runner: SubprocessRunner,
        model: str = "openai:gpt-4o",
        max_debug_rounds: int = 3,
    ) -> None:
        """Create a debugger agent.

        Args:
            runner: Sandbox runner for script execution.
            model: Pydantic AI model string.
            max_debug_rounds: Maximum repair attempts (default 3).
        """
        self.runner = runner
        self.max_debug_rounds = max_debug_rounds
        self.agent = Agent(
            model,
            name="debugger_agent",
            output_type=str,
            instructions=_DEBUGGER_INSTRUCTIONS,
            defer_model_check=True,
        )

    def debug(
        self,
        code: str,
        run_id: str = "debug",
        candidate_id: str = "debug",
        dataset_dir: str | None = None,
        dataset_files: list[str] | None = None,
    ) -> DebugOutcome:
        """Execute code and repair it until it runs or the budget is spent.

        Args:
            code: The Python script to execute and repair.
            run_id: Identifier of the current run.
            candidate_id: Identifier of the candidate.
            dataset_dir: Dataset directory to map into the sandbox.
            dataset_files: Dataset file names to map.

        Returns:
            A ``DebugOutcome`` describing the final code, execution result,
            and number of repair rounds.
        """
        sandbox = self.runner.prepare_sandbox(
            run_id=run_id,
            candidate_id=candidate_id,
            dataset_dir=dataset_dir,
            dataset_files=dataset_files,
        )
        result = self.runner.run_code(code, sandbox_dir=str(sandbox))
        current_code = code
        rounds = 0

        while self._needs_repair(result) and rounds < self.max_debug_rounds:
            rounds += 1
            with logfire.span("debugger.repair_round", round=rounds):
                error_text = result.stderr or result.stdout or f"exit code {result.returncode}"
                prompt = (
                    f"# Code with an error\n{current_code}\n"
                    f"# Error\n{error_text}\n"
                    f"# Your task\nPlease revise the code to fix the error."
                )
                try:
                    response = self.agent.run_sync(prompt)
                except Exception:
                    logfire.warn("debugger.repair_round.llm_failed", round=rounds)
                    continue
                current_code = response.output
            result = self.runner.run_code(current_code, sandbox_dir=str(sandbox))

        return DebugOutcome(
            code=current_code,
            result=result,
            debug_rounds=rounds,
            recovered=self._is_success(result),
        )

    @staticmethod
    def _needs_repair(result: ExecutionResult) -> bool:
        """Return whether the execution outcome requires repair."""
        return not result.success or result.validation_score is None

    @staticmethod
    def _is_success(result: ExecutionResult) -> bool:
        return result.success and result.validation_score is not None
