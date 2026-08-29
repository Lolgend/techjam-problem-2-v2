"""Data usage guardrail checker ($A_data$).

Cross-references the solution code with the task specification's dataset
metadata (Figure 22 prompt) to verify that every supplied dataset file is
consumed, and returns an improved script when information is left unused.
"""

from __future__ import annotations

import logfire
from pydantic_ai import Agent

from problem_2_v2.contracts.code_utils import extract_python_code
from problem_2_v2.contracts.guardrails import DataUsageStatus
from problem_2_v2.contracts.task import TaskSpecification

_USAGE_INSTRUCTIONS = (
    "You audit a Python solution for a machine learning task to check "
    "whether it uses all the information provided for training.\n"
    "# Your task\n"
    "- If the solution code does not use the information provided, try to "
    "incorporate all of it.\n"
    "- Do not bypass using try-except; just occur error so it can be "
    "debugged!\n"
    "- See the task description carefully, to know how to extract unused "
    "information effectively.\n"
    "- When improving the solution code by incorporating unused "
    "information, DO NOT FORGET to print out "
    "'Final Validation Performance: {final_validation_score}' as in the "
    "original solution code.\n"
    "# Response format\n"
    "- Option 1: If the code did not use all the provided information, "
    "respond with a single markdown code block (wrapped in ```) which is "
    "the improved code block. There should be no additional headings or "
    "text in your response.\n"
    "- Option 2: If the code used all the provided information, simply "
    "state that 'All the provided information is used.'"
)


class DataUsageCheckerAgent:
    """Verifies full dataset consumption and proposes improved code.

    Attributes:
        agent: Pydantic AI agent auditing solution code against the task
            description.
    """

    def __init__(self, model: str = "openai:gpt-4o") -> None:
        """Create a data usage checker.

        Args:
            model: Pydantic AI model string.
        """
        self.agent = Agent(
            model,
            name="data_usage_check_agent",
            output_type=str,
            instructions=_USAGE_INSTRUCTIONS,
            defer_model_check=True,
        )

    def audit(self, spec: TaskSpecification, code: str) -> DataUsageStatus:
        """Audit whether the solution consumes all provided dataset files.

        Args:
            spec: The task specification (dataset metadata source).
            code: The solution script to audit.

        Returns:
            A ``DataUsageStatus`` with the LLM verdict, any improved code
            block, and a deterministic missing-sources audit based on file
            references in the code.
        """
        missing = self._missing_sources(spec, code)

        with logfire.span("guardrails.usage_check"):
            prompt = (
                f"# Solution Code\n{code}\n"
                f"# Task description\n{spec.task_name}\n{spec.description}\n"
                f"Dataset files: {', '.join(spec.dataset_files)}\n"
                f"Target variable: {spec.target_variable}\n"
                f"# Your task\nCheck whether all provided information is "
                f"used, and improve the code if not."
            )
            response = self.agent.run_sync(prompt)
        output = response.output.strip()

        if self._all_used_statement(output):
            if missing:
                return DataUsageStatus(
                    all_data_used=False,
                    missing_sources=missing,
                    usage_recommendations=(
                        f"Dataset files not referenced by the solution: {', '.join(missing)}."
                    ),
                    improved_code_block=None,
                )
            return DataUsageStatus(
                all_data_used=True,
                missing_sources=[],
                usage_recommendations="All the provided information is used.",
                improved_code_block=None,
            )

        improved = extract_python_code(output)
        if improved:
            missing = self._missing_sources(spec, improved)
        return DataUsageStatus(
            all_data_used=False,
            missing_sources=missing,
            usage_recommendations="LLM proposed an improved solution incorporating unused data.",
            improved_code_block=improved or None,
        )

    @staticmethod
    def _all_used_statement(output: str) -> bool:
        """Detect the Figure 22 'all used' response marker."""
        return "all the provided information is used" in output.lower()

    @staticmethod
    def _missing_sources(spec: TaskSpecification, code: str) -> list[str]:
        """List dataset files that are never referenced by the code."""
        return [name for name in spec.dataset_files if name not in code]
