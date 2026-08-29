"""Unit tests for the candidate evaluation agent."""

import sys
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.enums import MetricDirection
from problem_2_v2.contracts.search import ModelCard
from problem_2_v2.contracts.task import TaskSpecification
from problem_2_v2.initialization.evaluator import (
    CandidateEvaluation,
    CandidateEvaluatorAgent,
)
from problem_2_v2.runner.debugger import DebuggerAgent
from problem_2_v2.runner.sandbox import SubprocessRunner

SCORE_CODE = "print('Final Validation Performance: 0.8123')"
FENCED_SCORE_CODE = f"```python\n{SCORE_CODE}\n```"


def _spec() -> TaskSpecification:
    return TaskSpecification.from_markdown(
        "**Task Name:** Demo\n"
        "**Task Type:** TABULAR_CLASSIFICATION\n"
        "**Metric Name:** AUROC\n"
        "**Metric Direction:** MAXIMIZE\n"
        "**Target Variable:** label\n"
        "**Baseline Score:** 0.80\n"
        "**Description:** classify.\n",
        dataset_dir="/data",
    )


def _card(name: str = "CatBoost") -> ModelCard:
    return ModelCard(
        model_name=name,
        rationale="state of the art",
        example_code="import catboost\nmodel = CatBoostClassifier()",
        library_dependencies=["catboost"],
    )


@pytest.fixture()
def runner(tmp_path: Path) -> SubprocessRunner:
    return SubprocessRunner(
        runs_dir=str(tmp_path / "runs"),
        timeout_seconds=5,
        python_executable=sys.executable,
    )


@pytest.fixture()
def debugger(runner: SubprocessRunner) -> DebuggerAgent:
    return DebuggerAgent(runner=runner, model="test", max_debug_rounds=2)


@pytest.fixture()
def evaluator(debugger: DebuggerAgent) -> CandidateEvaluatorAgent:
    return CandidateEvaluatorAgent(debugger=debugger, model="test")


class TestCandidateEvaluatorAgent:
    """Test candidate script generation and evaluation."""

    def test_evaluate_generates_and_runs_code(self, evaluator: CandidateEvaluatorAgent) -> None:
        with evaluator.agent.override(model=TestModel(custom_output_text=SCORE_CODE)):
            evaluation = evaluator.evaluate(_spec(), _card(), run_id="r", candidate_id="c")
        assert isinstance(evaluation, CandidateEvaluation)
        assert evaluation.score == pytest.approx(0.8123)
        assert evaluation.result.success is True
        assert evaluation.code == SCORE_CODE

    def test_evaluate_strips_markdown_fences(self, evaluator: CandidateEvaluatorAgent) -> None:
        with evaluator.agent.override(model=TestModel(custom_output_text=FENCED_SCORE_CODE)):
            evaluation = evaluator.evaluate(_spec(), _card(), run_id="r", candidate_id="c")
        assert "```" not in evaluation.code
        assert evaluation.score == pytest.approx(0.8123)

    def test_evaluate_marks_invalid_code_as_failed(
        self, evaluator: CandidateEvaluatorAgent
    ) -> None:
        with evaluator.agent.override(
            model=TestModel(custom_output_text="```python\nthis is not python (:\n```")
        ):
            evaluation = evaluator.evaluate(_spec(), _card(), run_id="r", candidate_id="c")
        assert evaluation.score is None
        assert evaluation.result.success is False

    def test_evaluate_all_returns_one_evaluation_per_card(
        self, evaluator: CandidateEvaluatorAgent
    ) -> None:
        cards = [_card("A"), _card("B"), _card("C")]
        with evaluator.agent.override(model=TestModel(custom_output_text=SCORE_CODE)):
            evaluations = evaluator.evaluate_all(_spec(), cards, run_id="r")
        assert len(evaluations) == 3
        assert all(e.score == pytest.approx(0.8123) for e in evaluations)

    def test_prompt_contains_model_and_task(self, evaluator: CandidateEvaluatorAgent) -> None:
        captured: dict[str, str] = {}

        def capturing_model(messages, info):
            captured["prompt"] = messages[-1].parts[0].content
            from pydantic_ai import ModelResponse, TextPart

            return ModelResponse(parts=[TextPart(content=SCORE_CODE)])

        from pydantic_ai.models.function import FunctionModel

        with evaluator.agent.override(model=FunctionModel(function=capturing_model)):
            evaluator.evaluate(_spec(), _card("LightGBM"), run_id="r", candidate_id="c")
        assert "LightGBM" in captured["prompt"]
        assert "AUROC" in captured["prompt"]
        assert "Final Validation Performance" in captured["prompt"]
        assert "30000" in captured["prompt"] or "30,000" in captured["prompt"]


class TestCandidateRanking:
    """Test descending permutation ranking under both directions."""

    @staticmethod
    def _evals() -> list[CandidateEvaluation]:
        base = {
            "result": None,
            "debug_rounds": 0,
        }
        return [
            CandidateEvaluation(model_name="m1", score=0.85, code="", **base),  # type: ignore[arg-type]
            CandidateEvaluation(model_name="m2", score=0.90, code="", **base),  # type: ignore[arg-type]
            CandidateEvaluation(model_name="m3", score=0.88, code="", **base),  # type: ignore[arg-type]
        ]

    def test_ranking_sorts_maximize_descending(self) -> None:
        ranked = CandidateEvaluatorAgent.ranking(self._evals(), MetricDirection.MAXIMIZE)
        assert [e.model_name for e in ranked] == ["m2", "m3", "m1"]

    def test_ranking_sorts_minimize_ascending(self) -> None:
        ranked = CandidateEvaluatorAgent.ranking(self._evals(), MetricDirection.MINIMIZE)
        assert [e.model_name for e in ranked] == ["m1", "m3", "m2"]

    def test_ranking_failed_evaluations_sort_last(self) -> None:
        evals = self._evals()
        evals.append(
            CandidateEvaluation(
                model_name="broken", score=None, code="", result=None, debug_rounds=0
            )
        )
        ranked = CandidateEvaluatorAgent.ranking(evals, MetricDirection.MAXIMIZE)
        assert ranked[-1].model_name == "broken"
