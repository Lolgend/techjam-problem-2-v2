"""Unit tests for the adaptive refinement planner agent."""

from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.enums import ComponentCategory
from problem_2_v2.contracts.refinement import RefinementPlan, TargetCodeBlock
from problem_2_v2.refinement.planner import RefinementPlannerAgent

TARGET_BLOCK = "model = LogisticRegression()\nmodel.fit(X, y)"
NEXT_PLAN_TEXT = (
    "Replace logistic regression with a ranker using pairwise loss, keeping the subsampling intact."
)


def _block() -> TargetCodeBlock:
    return TargetCodeBlock(
        raw_code=TARGET_BLOCK,
        category=ComponentCategory.MODEL_ARCHITECTURE,
        initial_plan="draft",
    )


class TestRefinementPlannerAgent:
    """Test Figure 16 history-conditioned plan generation."""

    def test_proposes_next_plan(self) -> None:
        agent = RefinementPlannerAgent(model="test")
        attempts = [("scale the features", 0.80)]
        with agent.agent.override(model=TestModel(custom_output_text=NEXT_PLAN_TEXT)):
            plan = agent.next_plan(_block(), attempts, iteration_index=1)
        assert isinstance(plan, RefinementPlan)
        assert plan.natural_language_plan == NEXT_PLAN_TEXT
        assert plan.iteration_index == 1
        assert plan.plan_id == "p1"

    def test_plan_id_increments_with_iteration(self) -> None:
        agent = RefinementPlannerAgent(model="test")
        with agent.agent.override(model=TestModel(custom_output_text="try interaction features")):
            plan = agent.next_plan(_block(), [], iteration_index=3)
        assert plan.plan_id == "p3"
        assert plan.iteration_index == 3

    def test_prompt_contains_attempt_history_with_scores(self) -> None:
        agent = RefinementPlannerAgent(model="test")
        captured: dict[str, str] = {}

        def capturing_model(messages, info):
            captured["prompt"] = messages[-1].parts[0].content
            return ModelResponse(parts=[TextPart(content="next plan")])

        attempts = [("plan one: scale features", 0.81), ("plan two: one-hot encode", 0.79)]
        with agent.agent.override(model=FunctionModel(function=capturing_model)):
            agent.next_plan(_block(), attempts, iteration_index=2)
        assert TARGET_BLOCK in captured["prompt"]
        assert "plan one: scale features" in captured["prompt"]
        assert "0.81" in captured["prompt"]
        assert "plan two: one-hot encode" in captured["prompt"]
        assert "0.79" in captured["prompt"]

    def test_prompt_handles_failed_attempts(self) -> None:
        agent = RefinementPlannerAgent(model="test")
        captured: dict[str, str] = {}

        def capturing_model(messages, info):
            captured["prompt"] = messages[-1].parts[0].content
            return ModelResponse(parts=[TextPart(content="next plan")])

        attempts = [("plan one", None)]
        with agent.agent.override(model=FunctionModel(function=capturing_model)):
            agent.next_plan(_block(), attempts, iteration_index=1)
        assert "N/A" in captured["prompt"] or "None" in captured["prompt"]
