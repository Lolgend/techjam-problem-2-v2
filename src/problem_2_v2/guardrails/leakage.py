"""Data leakage guardrail checker ($A_leakage$).

Inspects the solution's preprocessing blocks (Figure 20 prompt); when
leakage is detected, the suspicious block is repaired with the Figure 21
prompt and patched back into the solution prior to evaluation.
"""

from __future__ import annotations

import logfire
from pydantic_ai import Agent

from problem_2_v2.contracts.code_utils import extract_python_code
from problem_2_v2.contracts.guardrails import DataLeakageStatus
from problem_2_v2.contracts.refinement import align_replacement_indent, find_matching_block

_CHECK_PROMPT_TEMPLATE = (
    "# Python code\n"
    "{code}\n"
    "# Your task\n"
    "- Extract the code block where the validation and test samples are preprocessed using\n"
    "training samples.\n"
    "- Check that the model is trained with only training samples.\n"
    "- Check that before printing the final validation score, the model is not trained the\n"
    "validation samples.\n"
    "- Also check whether the validation and test samples are preprocessed correctly, preventing\n"
    "information from the validation or test samples from influencing the training process\n"
    "(i.e., preventing data leakage).\n"
    "# Requirement\n"
    "- Extract a code block and also check the data leakage.\n"
    "- The code block should be an exact subset of the above Python code.\n"
    "- Your response for a code block should be a single markdown code block.\n"
    "- If data leakage is present on validation and test samples, answer 'Yes Data Leakage'.\n"
    "- If data leakage is not present on validation and test samples, answer 'No Data Leakage'.\n"
    "Use this JSON schema:\n"
    "Answer = {{'leakage_status': str, 'code_block': str}}\n"
    "Return: list[Answer]"
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


class LeakageEnforcementError(RuntimeError):
    """Raised when strict leakage enforcement is enabled and repair fails.

    This error signals that data leakage was detected in the solution
    script and all repair attempts (including retries) were exhausted
    without resolving the leakage.
    """


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
            defer_model_check=True,
        )
        self.repair_agent = Agent(
            model,
            name="data_leakage_repair_agent",
            output_type=str,
            instructions=_REPAIR_INSTRUCTIONS,
            defer_model_check=True,
        )

    @staticmethod
    def build_check_prompt(code: str) -> str:
        """Build the Figure 21 data leakage check prompt.

        Args:
            code: The full solution script.

        Returns:
            The formatted prompt string.
        """
        return _CHECK_PROMPT_TEMPLATE.format(code=code)

    _build_check_prompt = build_check_prompt

    def check(self, code: str) -> DataLeakageStatus:
        """Inspect a solution script for data leakage.

        Args:
            code: The full solution script.

        Returns:
            A ``DataLeakageStatus`` describing the audit outcome and any
            suspicious preprocessing block.
        """
        with logfire.span("guardrails.leakage_check"):
            prompt = self.build_check_prompt(code)
            response = self.check_agent.run_sync(prompt)
        return response.output

    def repair(self, code: str, suspicious_block: str) -> str:
        """Repair a leaky preprocessing block and patch it into the script.

        Uses a multi-tier patching strategy (exact match → fuzzy match →
        full-script rewrite) so that formatting differences between the
        flagged block and the actual script never cause silent failures.

        Args:
            code: The full solution script.
            suspicious_block: The exact block flagged as leaky.

        Returns:
            The patched solution script with the corrected block, or the
            original script unchanged when all repair tiers are exhausted.
        """
        with logfire.span("guardrails.leakage_repair"):
            prompt = f"# Python code\n{code}\n# Your task\nRefine the code to prevent data leakage."
            response = self.repair_agent.run_sync(prompt)
        corrected = extract_python_code(response.output)
        if not corrected:
            logfire.warn("guardrails.leakage_repair.no_code")
            patched = self._full_script_rewrite(code)
        else:
            patched = self._patch(code, suspicious_block, corrected)
        if patched != code:
            logfire.info("guardrails.leakage_repair.succeeded")
        return patched

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
        repaired = self.repair(code, status.suspicious_code_block)
        return status, repaired

    def _patch(self, code: str, original: str, corrected: str) -> str:
        """Replace the original block with the corrected one using multi-tier matching.

        Tiers:
          1. Exact ``str.replace`` when the suspicious block is verbatim.
          2. Fuzzy match via ``find_matching_block()`` with whitespace
             normalization, quote unification, and AST fallback, then
             replace the matched segment with indentation-aligned correction.
          3. Full-script rewrite: ask the repair agent to rewrite the
             entire script with leakage fixed.

        Args:
            code: The full solution script.
            original: The suspicious block to locate.
            corrected: The corrected replacement block.

        Returns:
            The patched script. Falls back to the original code only when
            the full-script rewrite also fails to produce extractable code.
        """
        # Tier 1: exact string match.
        if original in code:
            return code.replace(original, corrected, 1)

        # Tier 2: fuzzy/normalized match via find_matching_block.
        matched = find_matching_block(original, code)
        if matched is not None:
            first_line = matched.splitlines()[0] if matched.splitlines() else ""
            indent = first_line[: len(first_line) - len(first_line.lstrip())]
            aligned = align_replacement_indent(corrected, indent)
            return code.replace(matched, aligned, 1)

        # Tier 3: full-script rewrite fallback.
        return self._full_script_rewrite(code)

    def _full_script_rewrite(self, code: str) -> str:
        """Ask the repair agent to rewrite the entire script with leakage fixed.

        This is the last-resort fallback when block-level patching fails
        because the suspicious block cannot be located in the script.

        Args:
            code: The full solution script.

        Returns:
            The rewritten script, or the original code unchanged when the
            repair agent produces no extractable Python code.
        """
        prompt = (
            "# Python code\n"
            f"{code}\n"
            "# Your task\n"
            "Rewrite the ENTIRE script below to prevent data leakage. "
            "Return the full corrected script as a single markdown code block."
        )
        with logfire.span("guardrails.leakage_repair.full_rewrite"):
            response = self.repair_agent.run_sync(prompt)
        rewritten = extract_python_code(response.output)
        if not rewritten:
            logfire.warn("guardrails.leakage_repair.failed")
            return code
        return rewritten
