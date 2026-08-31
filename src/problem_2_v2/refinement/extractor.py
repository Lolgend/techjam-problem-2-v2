"""Targeted code block extractor agent ($A_extractor$).

Analyzes the current solution together with the latest ablation summary
and previously refined blocks (Figure 14 prompt) to extract the exact code
block with the highest impact and produce the initial refinement plan
$p_0$.
"""

from __future__ import annotations

import ast

import logfire
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from problem_2_v2.contracts.enums import ComponentCategory
from problem_2_v2.contracts.refinement import RefinementPlan, TargetCodeBlock, find_matching_block

_COMPONENT_KEYWORDS = (
    "train",
    "model",
    "loss",
    "fit",
    "predict",
    "infer",
    "evaluate",
    "net",
    "network",
    "score",
)


def _component_score(name: str) -> int:
    """Score a definition name by how strongly it suggests a model component."""
    lowered = name.lower()
    return sum(1 for keyword in _COMPONENT_KEYWORDS if keyword in lowered)


def _fallback_primary_block(solution: str) -> str:
    """Extract the primary training/model/loss block from a solution.

    Uses AST to pick the top-level function/class definition with the
    strongest component-name signal (ties broken by source size). When the
    script has no definitions, the entire solution is returned so the
    refinement loop always has a target to refine.

    Args:
        solution: The current solution script.

    Returns:
        The verbatim source of the primary component block.
    """
    try:
        tree = ast.parse(solution)
    except SyntaxError:
        return solution
    candidates: list[tuple[int, int, ast.AST]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            segment = ast.get_source_segment(solution, node)
            if segment is None:
                continue
            candidates.append((_component_score(node.name), len(segment), node))
    if not candidates:
        return solution
    _, _, best = max(candidates, key=lambda item: (item[0], item[1]))
    segment = ast.get_source_segment(solution, best)
    return segment if segment is not None else solution


_EXTRACTOR_PROMPT_TEMPLATE = (
    "# Introduction\n"
    "- You are a Kaggle grandmaster attending a competition.\n"
    "- In order to win this competition, you need to extract a code block from the current\n"
    "Python solution and improve the extracted block for better performance.\n"
    "- Your suggestion should be based on the ablation study results of the current Python\n"
    "solution.\n"
    "- We will now provide the current Python solution and the ablation study results.\n"
    "- We also provide code blocks which you have tried to improve previously.\n"
    "# Python solution\n"
    "{solution_script}\n"
    "# Ablation study results\n"
    "{ablation_summary}\n"
    "{previous_blocks}"
    "# Your task\n"
    "- Given the ablation study results, suggest an effective next plan to improve the above\n"
    "Python script.\n"
    "- The plan should be a brief outline/sketch of your proposed solution in natural language\n"
    "(3-5 sentences).\n"
    "- Please avoid plan which can make the solution's running time too long (e.g., searching\n"
    "hyperparameters in a very large search space).\n"
    "- Try to improve the other part which was not considered before.\n"
    "- Also extract the code block from the above Python script that need to be improved\n"
    "according to the proposed plan. You should try to extract the code block which was not\n"
    "improved before.\n"
    "# Response format\n"
    "- Your response should be a brief outline/sketch of your proposed solution in natural\n"
    "language (3-5 sentences) and a single markdown code block which is the code block that need\n"
    "to be improved.\n"
    "- The code block can be long but should be exactly extracted from the Python script\n"
    "provided above.\n"
    "Use this JSON schema:\n"
    "Refine_Plan = {{'code_block': str, 'plan': str}}\n"
    "Return: list[Refine_Plan]"
)


class RefinePlanItem(BaseModel):
    """Structured extractor output: a target block plus its plan.

    Attributes:
        code_block: Exact code block extracted from the solution script.
        plan: Natural-language refinement plan (3-5 sentences).
        category: Functional category of the extracted block.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    code_block: str = Field(description="Exact extracted code block.")
    plan: str = Field(description="Natural-language refinement plan.")
    category: ComponentCategory = Field(
        default=ComponentCategory.MODEL_ARCHITECTURE,
        description="Component category.",
    )


class CodeBlockExtractorAgent:
    """Extracts the highest-impact code block and drafts the initial plan.

    Attributes:
        agent: Pydantic AI agent producing ``list[RefinePlanItem]`` output.
    """

    def __init__(
        self,
        model: str = "openai:gpt-4o",
        model_settings: ModelSettings | dict | None = None,
    ) -> None:
        """Create a code block extractor.

        Args:
            model: Pydantic AI model string.
            model_settings: Optional LLM generation settings (e.g. max_tokens).
        """
        self.model_settings = model_settings
        self.agent = Agent(
            model,
            name="code_block_extractor_agent",
            output_type=list[RefinePlanItem],
            model_settings=model_settings,
            defer_model_check=True,
        )

    def extract(
        self,
        solution: str,
        ablation_summary: str,
        previous_blocks: list[str],
    ) -> tuple[TargetCodeBlock, RefinementPlan]:
        """Extract the target code block and its initial refinement plan.

        Args:
            solution: The current solution script.
            ablation_summary: Summary of the latest ablation study.
            previous_blocks: Code blocks refined in previous outer loops.

        Returns:
            A ``(TargetCodeBlock, RefinementPlan)`` pair for the initial
            inner-loop iteration.

        Raises:
            ValueError: If the agent returns no refinement plans. A block
                that cannot be located verbatim in the solution is resolved
                through ``find_matching_block``'s resilient tiers and, as a
                last resort, AST primary-component extraction -- it never
                aborts the refinement loop.
        """
        prompt = self.build_prompt(solution, ablation_summary, previous_blocks)
        with logfire.span("extractor.llm"):
            response = self.agent.run_sync(prompt)
        items = response.output
        if not items:
            raise ValueError("Extractor returned no refinement plans.")
        item = items[0]

        matched = find_matching_block(item.code_block, solution)
        if matched is None:
            matched = _fallback_primary_block(solution)

        block = TargetCodeBlock(
            raw_code=matched,
            category=item.category,
            start_line=None,
            end_line=None,
            initial_plan=item.plan,
        )
        plan = RefinementPlan(
            plan_id="p0",
            natural_language_plan=item.plan,
            target_subcomponents=[],
            expected_gain="",
            iteration_index=0,
        )
        return block, plan

    @staticmethod
    def build_prompt(
        solution: str,
        ablation_summary: str,
        previous_blocks: list[str],
    ) -> str:
        """Build the Figure 14 extraction prompt.

        Args:
            solution: The current solution script.
            ablation_summary: Summary of the latest ablation study.
            previous_blocks: Code blocks refined in previous outer loops.

        Returns:
            The full extraction prompt.
        """
        history_parts = [
            f"## Code block {{{i}}}\n{block}" for i, block in enumerate(previous_blocks)
        ]
        history_joined = "\n".join(history_parts)
        previous_str = f"{history_joined}\n" if history_parts else ""
        return _EXTRACTOR_PROMPT_TEMPLATE.format(
            solution_script=solution,
            ablation_summary=ablation_summary,
            previous_blocks=previous_str,
        )

    _build_prompt = build_prompt
