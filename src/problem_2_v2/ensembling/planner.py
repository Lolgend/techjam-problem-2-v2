"""Adaptive ensemble planner agent ($A_ens_planner$).

Generates the initial ensemble strategy $e_0$ and proposes novel
subsequent strategies $e_r$ conditioned on the trajectory of attempted
ensemble plans and their scores (Figure 17 prompt), following Algorithm 3.
"""

from __future__ import annotations

import logfire
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent

from problem_2_v2.contracts.enums import EnsembleMethod
from problem_2_v2.contracts.guardrails import EnsembleStrategy
from problem_2_v2.contracts.task import PipelineArtifact

_PLANNER_INSTRUCTIONS = (
    "You are a Kaggle grandmaster attending a competition. In order to win "
    "this competition, you have to ensemble the provided Python Solutions "
    "for better performance.\n"
    "# Your task\n"
    "- Suggest a better plan to ensemble the solutions. Concentrate on how "
    "to merge, not on other parts like hyperparameters.\n"
    "- The suggested plan must be easy to implement, novel, and effective.\n"
    "- The suggested plan should differ from the previous plans you have "
    "tried and should receive a higher (or lower) score.\n"
    "- Plan should not modify the original solutions too much since "
    "execution error can occur.\n"
    "# Response format\n"
    "- Your response should be an outline/sketch of your proposed solution "
    "in natural language.\n"
    "- There should be no additional headings or text in your response."
)


class EnsemblePlanProposal(BaseModel):
    """Structured planner output describing the next ensemble strategy.

    Attributes:
        method: The ensembling technique to apply.
        natural_language_plan: Prose sketch of the ensembling plan.
        meta_learner_type: Meta-learner model class for stacking, if any.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    method: EnsembleMethod = Field(description="Ensembling technique.")
    natural_language_plan: str = Field(description="Plan sketch in prose.")
    meta_learner_type: str | None = Field(
        default=None,
        description="Meta-learner class for stacking.",
    )


class EnsemblePlannerAgent:
    """Proposes initial and adaptive ensemble strategies.

    Attributes:
        agent: Pydantic AI agent producing structured plan proposals.
    """

    def __init__(self, model: str = "openai:gpt-4o") -> None:
        """Create an ensemble planner.

        Args:
            model: Pydantic AI model string.
        """
        self.agent = Agent(
            model,
            name="ensemble_planner_agent",
            output_type=EnsemblePlanProposal,
            instructions=_PLANNER_INSTRUCTIONS,
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
                response = self.agent.run_sync(self._build_prompt(solutions, attempts=[]))
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
            response = self.agent.run_sync(self._build_prompt(solutions, attempts))
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
    def _build_prompt(
        solutions: list[PipelineArtifact],
        attempts: list[tuple[EnsembleStrategy, float | None]],
    ) -> str:
        """Build the Figure 17 planning prompt."""
        solution_blocks = "\n".join(
            f"# {index + 1}th Python Solution\n{artifact.full_code}"
            for index, artifact in enumerate(solutions)
        )
        history = "\n".join(
            f"## Plan: {strategy.natural_language_plan}\n## Score: "
            f"{score if score is not None else 'N/A'}"
            for strategy, score in attempts
        )
        return (
            f"# Introduction\nYou are a Kaggle grandmaster attending a "
            f"competition.\n{solution_blocks}\n"
            f"# Ensemble plans you have tried\n{history or '(none yet)'}\n"
            f"# Your task\nSuggest a better plan to ensemble the provided "
            f"solutions."
        )
