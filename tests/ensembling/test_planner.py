"""Unit tests for the adaptive ensemble planner agent."""

from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.enums import EnsembleMethod
from problem_2_v2.contracts.guardrails import EnsembleStrategy
from problem_2_v2.contracts.task import PipelineArtifact
from problem_2_v2.ensembling.planner import EnsemblePlannerAgent

SOLUTION_1 = "print('Final Validation Performance: 0.80')"
SOLUTION_2 = "print('Final Validation Performance: 0.82')"
PLAN_TEXT = (
    "Average the predicted probabilities from both models and threshold at "
    "0.5 to build the submission."
)


def _artifact(code: str, score: float, stage: str = "branch") -> PipelineArtifact:
    return PipelineArtifact(
        version=0,
        full_code=code,
        validation_score=score,
        parent_version=None,
        applied_diff=None,
        iteration_stage=stage,
    )


class TestEnsemblePlannerAgent:
    """Test Figure 17 initial and adaptive plan generation."""

    def _plan_args(self, method: str = "SIMPLE_AVERAGE") -> dict[str, object]:
        return {
            "method": method,
            "natural_language_plan": PLAN_TEXT,
            "meta_learner_type": None,
        }

    def test_generates_initial_plan(self) -> None:
        planner = EnsemblePlannerAgent(model="test")
        solutions = [
            _artifact(SOLUTION_1, 0.80, "branch_0"),
            _artifact(SOLUTION_2, 0.82, "branch_1"),
        ]
        with planner.agent.override(model=TestModel(custom_output_args=self._plan_args())):
            strategy = planner.initial_plan(solutions)
        assert isinstance(strategy, EnsembleStrategy)
        assert strategy.method is EnsembleMethod.SIMPLE_AVERAGE
        assert strategy.candidate_solution_ids == ["branch_0", "branch_1"]
        assert strategy.natural_language_plan == PLAN_TEXT

    def test_proposes_novel_plan_from_history(self) -> None:
        planner = EnsemblePlannerAgent(model="test")
        solutions = [
            _artifact(SOLUTION_1, 0.80, "branch_0"),
            _artifact(SOLUTION_2, 0.82, "branch_1"),
        ]
        attempts = [
            (
                EnsembleStrategy(
                    method=EnsembleMethod.SIMPLE_AVERAGE,
                    natural_language_plan="average probabilities",
                    meta_learner_type=None,
                    candidate_solution_ids=["branch_0", "branch_1"],
                    code_template=None,
                ),
                0.81,
            )
        ]
        with planner.agent.override(
            model=TestModel(custom_output_args=self._plan_args(method="STACKING_META_LEARNER"))
        ):
            strategy = planner.next_plan(solutions, attempts, iteration_index=1)
        assert strategy.method is EnsembleMethod.STACKING_META_LEARNER
        assert strategy.candidate_solution_ids == ["branch_0", "branch_1"]

    def test_prompt_contains_solutions_and_history(self) -> None:
        planner = EnsemblePlannerAgent(model="test")
        captured: dict[str, str] = {}

        def capturing_model(messages, info):
            captured["prompt"] = messages[-1].parts[0].content
            return ModelResponse(
                parts=[
                    TextPart(
                        content='{"method": "SIMPLE_AVERAGE", '
                        '"natural_language_plan": "average", '
                        '"meta_learner_type": null}'
                    )
                ]
            )

        solutions = [_artifact(SOLUTION_1, 0.80)]
        attempts = [
            (
                EnsembleStrategy(
                    method=EnsembleMethod.SIMPLE_AVERAGE,
                    natural_language_plan="average probabilities",
                    meta_learner_type=None,
                    candidate_solution_ids=["branch_0"],
                    code_template=None,
                ),
                0.81,
            )
        ]
        with planner.agent.override(model=FunctionModel(function=capturing_model)):
            planner.next_plan(solutions, attempts, iteration_index=1)
        assert "Final Validation Performance: 0.80" in captured["prompt"]
        assert "average probabilities" in captured["prompt"]
        assert "0.81" in captured["prompt"]

    def test_initial_plan_falls_back_to_simple_average(self) -> None:
        planner = EnsemblePlannerAgent(model="test")

        def exploding_model(messages, info):
            raise RuntimeError("LLM backend down")

        solutions = [_artifact(SOLUTION_1, 0.80, "branch_0")]
        with planner.agent.override(model=FunctionModel(function=exploding_model)):
            strategy = planner.initial_plan(solutions)
        assert strategy.method is EnsembleMethod.SIMPLE_AVERAGE
        assert strategy.candidate_solution_ids == ["branch_0"]
