"""Adaptive refinement planner agent ($A_planner$).

Proposes subsequent novel refinement plans $p_k$ for the inner loop,
conditioned on the trajectory of previously attempted plans and their
validation scores (Figure 16 prompt).
"""

from __future__ import annotations

import logfire
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from problem_2_v2.contracts.refinement import RefinementPlan, TargetCodeBlock

_PLANNER_PROMPT_TEMPLATE = (
    "# Introduction\n"
    "- You are a Kaggle grandmaster attending a competition.\n"
    "- In order to win this competition, you have to improve the code block for better\n"
    "performance.\n"
    "- We will provide the code block you are improving and the improvement plans you have\n"
    "tried.\n"
    "# Code block\n"
    "{code_block}\n"
    "# Improvement plans you have tried\n"
    "{history}"
    "# Your task\n"
    "- Suggest a better plan to improve the above code block.\n"
    "- The suggested plan must be novel and effective.\n"
    "- Please avoid plans which can make the solution's running time too long (e.g., searching\n"
    "hyperparameters in a very large search space).\n"
    "- The suggested plan should be differ from the previous plans you have tried and should\n"
    "receive a higher score.\n"
    "# Response format\n"
    "- Your response should be a brief outline/sketch of your proposed solution in natural\n"
    "language (3-5 sentences).\n"
    "- There should be no additional headings or text in your response."
)


class RefinementPlannerAgent:
    """Generates adaptive refinement plans from attempt history.

    Attributes:
        agent: Pydantic AI agent producing the next plan in prose.
    """

    def __init__(
        self,
        model: str = "openai:gpt-4o",
        model_settings: ModelSettings | dict | None = None,
    ) -> None:
        """Create a refinement planner.

        Args:
            model: Pydantic AI model string.
            model_settings: Optional LLM generation settings (e.g. max_tokens).
        """
        self.model_settings = model_settings
        self.agent = Agent(
            model,
            name="refinement_planner_agent",
            output_type=str,
            model_settings=model_settings,
            defer_model_check=True,
        )

    @staticmethod
    def build_prompt(code_block: str, attempts: list[tuple[str, float | None]]) -> str:
        """Build the Figure 16 refinement planning prompt.

        Args:
            code_block: The code block being refined.
            attempts: Previously attempted plans paired with their validation scores.

        Returns:
            The formatted planning prompt string.
        """
        history_parts = [
            f"## Plan: {plan}\n## Score: {score if score is not None else 'N/A'}"
            for plan, score in attempts
        ]
        history_joined = "\n".join(history_parts)
        history_str = f"{history_joined}\n" if history_parts else ""
        return _PLANNER_PROMPT_TEMPLATE.format(
            code_block=code_block,
            history=history_str,
        )

    _build_prompt = build_prompt

    def next_plan(
        self,
        target_block: TargetCodeBlock,
        attempts: list[tuple[str, float | None]],
        iteration_index: int,
    ) -> RefinementPlan:
        """Propose the next refinement plan for the inner loop.

        Args:
            target_block: The code block being refined.
            attempts: Previously attempted plans paired with their
                validation scores (``None`` marks a failed attempt).
            iteration_index: Inner-loop iteration index (k).

        Returns:
            A ``RefinementPlan`` with a fresh plan id and the proposed
            strategy.
        """
        prompt = self.build_prompt(target_block.raw_code, attempts)
        with logfire.span("planner.next_plan", iteration=iteration_index):
            response = self.agent.run_sync(prompt)
        return RefinementPlan(
            plan_id=f"p{iteration_index}",
            natural_language_plan=response.output,
            target_subcomponents=[],
            expected_gain="",
            iteration_index=iteration_index,
        )
