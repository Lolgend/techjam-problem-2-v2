"""Unit tests for the greedy sequential model merger."""

import sys
from pathlib import Path

import pytest
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.enums import MetricDirection
from problem_2_v2.contracts.task import TaskSpecification
from problem_2_v2.initialization.evaluator import CandidateEvaluation
from problem_2_v2.initialization.merger import MergeOutcome, ModelMergerAgent
from problem_2_v2.runner.debugger import DebuggerAgent
from problem_2_v2.runner.sandbox import SubprocessRunner

CODE_A = "print('Final Validation Performance: 0.90')\n# candidate A"
CODE_B = "print('Final Validation Performance: 0.95')\n# candidate B"
CODE_C = "print('Final Validation Performance: 0.85')\n# candidate C"


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


def _evaluation(model_name: str, code: str, score: float) -> CandidateEvaluation:
    return CandidateEvaluation(model_name=model_name, code=code, debug_rounds=0, score=score)


def scripted_merger_model(scripted: dict[str, float]):
    """Return a FunctionModel that maps reference code to a merged score."""

    def fn(messages, info):
        prompt = messages[-1].parts[0].content
        for ref_code, score in scripted.items():
            if ref_code in prompt:
                return ModelResponse(
                    parts=[TextPart(content=f"print('Final Validation Performance: {score}')")]
                )
        return ModelResponse(parts=[TextPart(content="print('Final Validation Performance: 0.0')")])

    return FunctionModel(function=fn)


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
def merger(debugger: DebuggerAgent) -> ModelMergerAgent:
    return ModelMergerAgent(debugger=debugger, model="test")


class TestModelMergerAgent:
    """Test Algorithm 1 greedy merging behavior."""

    def test_accepts_improvement_and_stops_on_regression(self, merger: ModelMergerAgent) -> None:
        ranked = [
            _evaluation("A", CODE_A, 0.90),
            _evaluation("B", CODE_B, 0.95),
            _evaluation("C", CODE_C, 0.85),
        ]
        scripted = {CODE_B: 0.95, CODE_C: 0.85}
        with merger.agent.override(model=scripted_merger_model(scripted)):
            outcome = merger.merge(_spec(), ranked, run_id="r")
        assert isinstance(outcome, MergeOutcome)
        assert outcome.final_score == pytest.approx(0.95)
        assert outcome.merged_count == 1
        assert len(outcome.steps) == 2
        assert outcome.steps[0].accepted is True
        assert outcome.steps[1].accepted is False
        assert outcome.steps[1].reason == "rejected_score"

    def test_accepts_equal_score(self, merger: ModelMergerAgent) -> None:
        ranked = [
            _evaluation("A", CODE_A, 0.90),
            _evaluation("B", CODE_B, 0.90),
        ]
        scripted = {CODE_B: 0.90}
        with merger.agent.override(model=scripted_merger_model(scripted)):
            outcome = merger.merge(_spec(), ranked, run_id="r")
        assert outcome.final_score == pytest.approx(0.90)
        assert outcome.merged_count == 1

    def test_minimize_direction_accepts_lower_scores(self, merger: ModelMergerAgent) -> None:
        spec = _spec()
        spec.metric_direction = MetricDirection.MINIMIZE
        ranked = [
            _evaluation("A", CODE_A, 0.90),
            _evaluation("B", CODE_B, 0.70),
        ]
        scripted = {CODE_B: 0.70}
        with merger.agent.override(model=scripted_merger_model(scripted)):
            outcome = merger.merge(spec, ranked, run_id="r")
        assert outcome.final_score == pytest.approx(0.70)
        assert outcome.merged_count == 1

    def test_stops_on_merge_failure(self, merger: ModelMergerAgent) -> None:
        ranked = [
            _evaluation("A", CODE_A, 0.90),
            _evaluation("B", CODE_B, 0.95),
        ]
        with (
            merger.agent.override(model=TestModel(custom_output_text="def broken(:\n    pass")),
            merger.debugger.agent.override(
                model=TestModel(custom_output_text="def still_broken(:")
            ),
        ):
            outcome = merger.merge(_spec(), ranked, run_id="r")
        assert outcome.merged_count == 0
        assert outcome.final_code == CODE_A
        assert outcome.final_score == pytest.approx(0.90)
        assert outcome.steps[0].reason == "rejected_error"

    def test_single_candidate_has_no_merge_steps(self, merger: ModelMergerAgent) -> None:
        ranked = [_evaluation("A", CODE_A, 0.90)]
        outcome = merger.merge(_spec(), ranked, run_id="r")
        assert outcome.merged_count == 0
        assert outcome.final_code == CODE_A
        assert outcome.final_score == pytest.approx(0.90)

    def test_empty_rankings_produce_empty_outcome(self, merger: ModelMergerAgent) -> None:
        outcome = merger.merge(_spec(), [], run_id="r")
        assert outcome.final_code == ""
        assert outcome.final_score is None
        assert outcome.merged_count == 0

    def test_lineage_links_accepted_merges(self, merger: ModelMergerAgent) -> None:
        ranked = [
            _evaluation("A", CODE_A, 0.90),
            _evaluation("B", CODE_B, 0.95),
        ]
        scripted = {CODE_B: 0.95}
        with merger.agent.override(model=scripted_merger_model(scripted)):
            outcome = merger.merge(_spec(), ranked, run_id="r")
        assert len(outcome.lineage) == 2
        assert outcome.lineage[0].version == 0
        assert outcome.lineage[0].parent_version is None
        assert outcome.lineage[1].version == 1
        assert outcome.lineage[1].parent_version == 0
        assert outcome.lineage[1].applied_diff is not None
