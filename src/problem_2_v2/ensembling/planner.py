"""Adaptive ensemble planner agent ($A_ens_planner$).

Generates the initial ensemble strategy $e_0$ and proposes novel
subsequent strategies $e_r$ conditioned on the trajectory of attempted
ensemble plans and their scores (Figure 17 prompt), following Algorithm 3.
"""

from __future__ import annotations

import logfire
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from problem_2_v2.contracts.enums import EnsembleMethod
from problem_2_v2.contracts.guardrails import EnsembleStrategy
from problem_2_v2.contracts.task import PipelineArtifact

_PLANNER_PROMPT_TEMPLATE = (
    "# Introduction\n"
    "- You are a Kaggle grandmaster attending a competition.\n"
    "- In order to win this competition, you have to ensemble {L} Python Solutions for better\n"
    "performance.\n"
    "- We will provide the Python Solutions and the ensemble plans you have tried.\n"
    "{solutions_block}\n"
    "# Ensemble plans you have tried\n"
    "{history}"
    "# Your task\n"
    "- Suggest a better plan to ensemble the {L} solutions. You should concentrate how to merge,\n"
    "not the other parts like hyperparameters.\n"
    "- The suggested plan must be easy to implement, novel, and effective.\n"
    "- The suggested plan should be differ from the previous plans you have tried and should\n"
    "receive a higher (or lower) score.\n"
    "# Response format\n"
    "- Your response should be an outline/sketch of your proposed solution in natural language.\n"
    "- There should be no additional headings or text in your response.\n"
    "- Plan should not modify the original solutions too much since execution error can occur."
)


class EnsemblePlanProposal(BaseModel):
    """Structured planner output describing the next ensemble strategy.

    Attributes:
        natural_language_plan: Prose sketch of the ensembling plan.
        method: The ensembling technique to apply.
        meta_learner_type: Meta-learner model class for stacking, if any.
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    natural_language_plan: str = Field(description="Plan sketch in prose.")
    method: EnsembleMethod = Field(
        default=EnsembleMethod.SIMPLE_AVERAGE,
        description="Ensembling technique.",
    )
    meta_learner_type: str | None = Field(
        default=None,
        description="Meta-learner class for stacking.",
    )


class EnsemblePlannerAgent:
    """Proposes initial and adaptive ensemble strategies.

    Attributes:
        agent: Pydantic AI agent producing structured plan proposals.
    """

    def __init__(
        self,
        model: str = "openai:gpt-4o",
        model_settings: ModelSettings | dict | None = None,
    ) -> None:
        """Create an ensemble planner.

        Args:
            model: Pydantic AI model string.
            model_settings: Optional LLM generation settings (e.g. max_tokens).
        """
        self.model_settings = model_settings
        self.agent = Agent(
            model,
            name="ensemble_planner_agent",
            output_type=EnsemblePlanProposal,
            model_settings=model_settings,
            defer_model_check=True,
        )

    def initial_plan(self, solutions: list[PipelineArtifact]) -> EnsembleStrategy:
        """Generate the initial ensemble plan $e_0$.

        Args:
            solutions: The candidate solution artifacts.

        Returns:
            The initial ``EnsembleStrategy``. Falls back to a deterministic
            simple-average strategy when the LLM is unavailable.
        """
        try:
            with logfire.span("ens_planner.initial"):
                prompt = self.build_prompt(solutions, attempts=[])
                response = self.agent.run_sync(prompt)
            return self._to_strategy(response.output, solutions)
        except Exception:
            logfire.warn("ens_planner.initial.failed; using simple-average fallback")
            return self._default_initial_plan(solutions)

    def next_plan(
        self,
        solutions: list[PipelineArtifact],
        attempts: list[tuple[EnsembleStrategy, float | None]],
        iteration_index: int,
    ) -> EnsembleStrategy:
        """Propose the next novel ensemble plan $e_r$.

        Args:
            solutions: The candidate solution artifacts.
            attempts: Previously attempted strategies paired with their
                validation scores (``None`` marks a failed attempt).
            iteration_index: Ensemble round index (r).

        Returns:
            The next ``EnsembleStrategy``.
        """
        with logfire.span("ens_planner.next", round=iteration_index):
            prompt = self.build_prompt(solutions, attempts)
            response = self.agent.run_sync(prompt)
        return self._to_strategy(response.output, solutions)

    @staticmethod
    def _to_strategy(
        proposal: EnsemblePlanProposal, solutions: list[PipelineArtifact]
    ) -> EnsembleStrategy:
        """Build an ``EnsembleStrategy`` from the proposal."""
        return EnsembleStrategy(
            method=proposal.method,
            natural_language_plan=proposal.natural_language_plan,
            meta_learner_type=proposal.meta_learner_type,
            candidate_solution_ids=[artifact.iteration_stage for artifact in solutions],
            code_template=None,
        )

    @staticmethod
    def _default_initial_plan(solutions: list[PipelineArtifact]) -> EnsembleStrategy:
        """Deterministic fallback: simple probability averaging."""
        return EnsembleStrategy(
            method=EnsembleMethod.SIMPLE_AVERAGE,
            natural_language_plan=(
                "Average the predicted probabilities from all candidate "
                "solutions and produce the submission from the averaged "
                "predictions."
            ),
            meta_learner_type=None,
            candidate_solution_ids=[artifact.iteration_stage for artifact in solutions],
            code_template=None,
        )

    @staticmethod
    def build_prompt(
        solutions: list[PipelineArtifact],
        attempts: list[tuple[EnsembleStrategy, float | None]],
    ) -> str:
        """Build the Figure 19 ensemble strategy planning prompt.

        Args:
            solutions: The candidate solution artifacts.
            attempts: Previously attempted strategies paired with their validation scores.

        Returns:
            The formatted ensemble planning prompt string.
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

        history_parts = [
            f"## Plan: {strategy.natural_language_plan}\n## Score: "
            f"{score if score is not None else 'N/A'}"
            for strategy, score in attempts
        ]
        history_joined = "\n".join(history_parts)
        history_str = f"{history_joined}\n" if history_parts else ""

        return _PLANNER_PROMPT_TEMPLATE.format(
            L=len(solutions),
            solutions_block=solutions_block,
            history=history_str,
        )

    _build_prompt = build_prompt
