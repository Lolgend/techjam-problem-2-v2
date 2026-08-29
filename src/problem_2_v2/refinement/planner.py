"""Adaptive refinement planner agent ($A_planner$).

Proposes subsequent novel refinement plans $p_k$ for the inner loop,
conditioned on the trajectory of previously attempted plans and their
validation scores (Figure 16 prompt).
"""

from __future__ import annotations

import logfire
from pydantic_ai import Agent

from problem_2_v2.contracts.refinement import RefinementPlan, TargetCodeBlock

_PLANNER_INSTRUCTIONS = (
    "You are a Kaggle grandmaster attending a competition. In order to win "
    "this competition, you have to improve the code block for better "
    "performance.\n"
    "# Your task\n"
    "- Suggest a better plan to improve the provided code block.\n"
    "- The suggested plan must be novel and effective.\n"
    "- Avoid plans which can make the solution's running time too long "
    "(e.g., searching hyperparameters in a very large search space).\n"
    "- The suggested plan should differ from the previous plans you have "
    "tried and should receive a higher score.\n"
    "# Response format\n"
    "- Your response should be a brief outline/sketch of your proposed "
    "solution in natural language (3-5 sentences).\n"
    "- There should be no additional headings or text in your response."
)


class RefinementPlannerAgent:
    """Generates adaptive refinement plans from attempt history.

    Attributes:
        agent: Pydantic AI agent producing the next plan in prose.
    """

    def __init__(self, model: str = "openai:gpt-4o") -> None:
        """Create a refinement planner.

        Args:
            model: Pydantic AI model string.
        """
        self.agent = Agent(
            model,
            name="refinement_planner_agent",
            output_type=str,
            instructions=_PLANNER_INSTRUCTIONS,
            defer_model_check=True,
        )

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
        history = "\n".join(
            f"## Plan: {plan}\n## Score: {score if score is not None else 'N/A'}"
            for plan, score in attempts
        )
        prompt = (
            f"# Code block\n{target_block.raw_code}\n"
            f"# Improvement plans you have tried\n{history or '(none yet)'}\n"
            f"# Your task\nSuggest a better, novel plan to improve the "
            f"above code block."
        )
        with logfire.span("planner.next_plan", iteration=iteration_index):
            response = self.agent.run_sync(prompt)
        return RefinementPlan(
            plan_id=f"p{iteration_index}",
            natural_language_plan=response.output,
            target_subcomponents=[],
            expected_gain="",
            iteration_index=iteration_index,
        )
