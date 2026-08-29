"""Data leakage guardrail checker ($A_leakage$).

Inspects the solution's preprocessing blocks (Figure 20 prompt); when
leakage is detected, the suspicious block is repaired with the Figure 21
prompt and patched back into the solution prior to evaluation.
"""

from __future__ import annotations

import logfire
from pydantic_ai import Agent

from problem_2_v2.contracts.enums import ComponentCategory
from problem_2_v2.contracts.guardrails import DataLeakageStatus
from problem_2_v2.contracts.refinement import TargetCodeBlock, block_in_script

_CHECK_INSTRUCTIONS = (
    "You audit Python machine learning code for data leakage.\n"
    "# Your task\n"
    "- Extract the code block where the validation and test samples are "
    "preprocessed using training samples.\n"
    "- Check that the model is trained with only training samples.\n"
    "- Check that before printing the final validation score, the model is "
    "not trained on the validation samples.\n"
    "- Check whether the validation and test samples are preprocessed "
    "correctly, preventing information from the validation or test samples "
    "from influencing the training process (i.e., preventing data "
    "leakage).\n"
    "# Requirement\n"
    "- Extract a code block and also check the data leakage.\n"
    "- The code block should be an exact subset of the provided Python "
    "code.\n"
    "- If data leakage is present on validation and test samples, set the "
    "status to 'Yes Data Leakage'; otherwise 'No Data Leakage'."
)

_REPAIR_INSTRUCTIONS = (
    "You repair data leakage in Python machine learning code.\n"
    "# Your task\n"
    "- In the provided Python code, the validation and test samples are "
    "influencing the training process, i.e., not correctly preprocessed.\n"
    "- Ensure that the model is trained with only training samples.\n"
    "- Ensure that before printing the final validation score, the model is "
    "not trained on the validation samples.\n"
    "- Refine the code to prevent such data leakage problem.\n"
    "# Requirement\n"
    "- Your response should be a single markdown code block.\n"
    "- Note that all the variables are defined earlier. Just modify it with "
    "the above code."
)


class DataLeakageCheckerAgent:
    """Detects and repairs data leakage in solution scripts.

    Attributes:
        check_agent: Pydantic AI agent producing ``DataLeakageStatus``.
        repair_agent: Pydantic AI agent producing corrected code blocks.
    """

    def __init__(self, model: str = "openai:gpt-4o") -> None:
        """Create a leakage checker.

        Args:
            model: Pydantic AI model string.
        """
        self.check_agent = Agent(
            model,
            name="data_leakage_check_agent",
            output_type=DataLeakageStatus,
            instructions=_CHECK_INSTRUCTIONS,
            defer_model_check=True,
        )
        self.repair_agent = Agent(
            model,
            name="data_leakage_repair_agent",
            output_type=str,
            instructions=_REPAIR_INSTRUCTIONS,
            defer_model_check=True,
        )

    def check(self, code: str) -> DataLeakageStatus:
        """Inspect a solution script for data leakage.

        Args:
            code: The full solution script.

        Returns:
            A ``DataLeakageStatus`` describing the audit outcome and any
            suspicious preprocessing block.
        """
        with logfire.span("guardrails.leakage_check"):
            prompt = f"# Python code\n{code}\n# Your task\nCheck for data leakage."
            response = self.check_agent.run_sync(prompt)
        return response.output

    def repair(self, code: str, suspicious_block: str) -> str:
        """Repair a leaky preprocessing block and patch it into the script.

        Args:
            code: The full solution script.
            suspicious_block: The exact block flagged as leaky.

        Returns:
            The patched solution script with the corrected block.

        Raises:
            ValueError: If the corrected block cannot be located in the
                solution (e.g. the flagged block is stale).
        """
        with logfire.span("guardrails.leakage_repair"):
            prompt = f"# Python code\n{code}\n# Your task\nRefine the code to prevent data leakage."
            response = self.repair_agent.run_sync(prompt)
        corrected = response.output.strip()
        return self._patch(code, suspicious_block, corrected)

    def audit(self, code: str) -> tuple[DataLeakageStatus, str]:
        """Check for leakage and repair the script if needed.

        Args:
            code: The full solution script.

        Returns:
            A ``(status, code)`` pair where ``code`` is the original script
            when clean, or the repaired script when leakage was detected
            and corrected.
        """
        status = self.check(code)
        if not status.is_leaking or not status.suspicious_code_block:
            return status, code
        try:
            repaired = self.repair(code, status.suspicious_code_block)
        except ValueError:
            logfire.warn("guardrails.leakage_repair.failed")
            return status, code
        return status, repaired

    @staticmethod
    def _patch(code: str, original: str, corrected: str) -> str:
        """Replace the original block with the corrected one, AST-safe."""
        if original in code:
            return code.replace(original, corrected, 1)

        if not block_in_script(original, code):
            raise ValueError("Suspicious block not found in solution script.")
        target = TargetCodeBlock(
            raw_code=original,
            category=ComponentCategory.DATA_PREPROCESSING,
            initial_plan="",
        )
        return target.replace_in(code, corrected)
