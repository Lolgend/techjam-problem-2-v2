"""Coder agent ($A_coder$) and AST-safe script patching.

``CoderAgent`` transforms an extracted target block according to the
active refinement plan (Figure 15 prompt). ``patch_script`` applies the
refined block to the full solution with an AST syntax check and an
indentation-tolerant whitespace-normalized fallback.
"""

from __future__ import annotations

import logfire
from pydantic_ai import Agent

from problem_2_v2.contracts.code_utils import extract_python_code, validate_python_syntax
from problem_2_v2.contracts.enums import ComponentCategory
from problem_2_v2.contracts.refinement import RefinementPlan, TargetCodeBlock

_CODER_INSTRUCTIONS = (
    "You are a Kaggle grandmaster attending a competition. In order to win "
    "this competition, you need to refine the code block for better "
    "performance based on the improvement plan.\n"
    "# Your task\n"
    "- Implement the improvement plan on the provided code block. But do "
    "not remove subsampling if exists.\n"
    "- The code block should be improved according to the proposed plan.\n"
    "- Note that all the variables including actual data are defined "
    "earlier (since you are just seeing a code block), therefore do not "
    "introduce dummy variables.\n"
    "# Response format\n"
    "- Your response should be a single markdown code block (wrapped in "
    "```) which is the improved code block.\n"
    "- There should be no additional headings or text in your response."
)


def patch_script(script: str, target_code: str, replacement: str) -> str:
    """Replace ``target_code`` with ``replacement`` inside ``script``.

    Tries an exact substring replacement first; when the block was
    extracted without its surrounding indentation, falls back to the
    indentation-tolerant ``TargetCodeBlock.replace_in`` matcher. The
    resulting script must parse as valid Python.

    Args:
        script: The full solution script.
        target_code: The code block to replace.
        replacement: The refined code block.

    Returns:
        The patched solution script.

    Raises:
        ValueError: If the block cannot be located or the patched script
            is not valid Python.
    """
    if target_code in script:
        candidate = script.replace(target_code, replacement, 1)
    else:
        block = TargetCodeBlock(
            raw_code=target_code,
            category=ComponentCategory.MODEL_ARCHITECTURE,
            start_line=None,
            end_line=None,
            initial_plan="",
        )
        candidate = block.replace_in(script, replacement)

    valid, error = validate_python_syntax(candidate)
    if not valid:
        raise ValueError(f"Patched script is invalid Python: {error}")
    return candidate


class CoderAgent:
    """Transforms an extracted target block into a refined block.

    Attributes:
        agent: Pydantic AI agent producing the refined code block.
    """

    def __init__(self, model: str = "openai:gpt-4o") -> None:
        """Create a coder agent.

        Args:
            model: Pydantic AI model string.
        """
        self.agent = Agent(
            model,
            name="coder_agent",
            output_type=str,
            instructions=_CODER_INSTRUCTIONS,
            defer_model_check=True,
        )

    def refine(self, target_block: TargetCodeBlock, plan: RefinementPlan) -> str:
        """Produce the refined version of the target code block.

        Args:
            target_block: The extracted code block to refine.
            plan: The refinement plan to implement.

        Returns:
            The cleaned refined code block (markdown fences stripped).
        """
        prompt = (
            f"# Code block\n{target_block.raw_code}\n"
            f"# Improvement plan\n{plan.natural_language_plan}\n"
            f"# Your task\nImplement the improvement plan on the above code "
            f"block. Do not remove subsampling if exists."
        )
        with logfire.span("coder.refine", plan_id=plan.plan_id):
            response = self.agent.run_sync(prompt)
        return extract_python_code(response.output)
