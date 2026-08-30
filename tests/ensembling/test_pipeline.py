"""Unit tests for the iterative ensemble pipeline."""

import json
import sys
from pathlib import Path

import pytest
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.iteration import IterationLogEntry
from problem_2_v2.contracts.task import PipelineArtifact, TaskSpecification
from problem_2_v2.ensembling.ensembler import EnsemblerAgent
from problem_2_v2.ensembling.pipeline import (
    EnsemblePipeline,
    EnsembleResult,
)
from problem_2_v2.ensembling.planner import EnsemblePlannerAgent
from problem_2_v2.runner.debugger import DebuggerAgent
from problem_2_v2.runner.sandbox import SubprocessRunner


def _spec() -> TaskSpecification:
    return TaskSpecification.from_markdown(
        "**Task Type:** TABULAR_CLASSIFICATION\n"
        "**Metric Name:** AUROC\n"
        "**Metric Direction:** MAXIMIZE\n"
        "**Target Variable:** label\n"
        "**Dataset Files:** train.csv\n",
        dataset_dir="/data",
    )


def _artifact(code: str, score: float, stage: str) -> PipelineArtifact:
    return PipelineArtifact(
        version=0,
        full_code=code,
        validation_score=score,
        parent_version=None,
        applied_diff=None,
        iteration_stage=stage,
    )


def _solutions() -> list[PipelineArtifact]:
    return [
        _artifact("print('Final Validation Performance: 0.80')", 0.80, "branch_0"),
        _artifact("print('Final Validation Performance: 0.82')", 0.82, "branch_1"),
    ]


def scripted_ensembler_model(plan_to_score: dict[str, float]):
    """Return a FunctionModel mapping plan text to a merged score."""

    def fn(messages, info):
        prompt = messages[-1].parts[0].content
        for plan, score in plan_to_score.items():
            if plan in prompt:
                return ModelResponse(
                    parts=[TextPart(content=f"print('Final Validation Performance: {score}')")]
                )
        return ModelResponse(parts=[TextPart(content="print('Final Validation Performance: 0.0')")])

    return FunctionModel(function=fn)


def _planner_args(method: str, plan: str) -> dict[str, object]:
    return {
        "method": method,
        "natural_language_plan": plan,
        "meta_learner_type": None,
    }


def scripted_planner_model(plan0: str, plan1: str):
    """Return a FunctionModel returning plan0 on the first call, plan1 after."""

    calls = {"count": 0}

    def fn(messages, info):
        calls["count"] += 1
        if calls["count"] == 1:
            return ModelResponse(
                parts=[TextPart(content=json.dumps(_planner_args("SIMPLE_AVERAGE", plan0)))]
            )
        return ModelResponse(
            parts=[TextPart(content=json.dumps(_planner_args("STACKING_META_LEARNER", plan1)))]
        )

    return FunctionModel(function=fn)


@pytest.fixture()
def runner(tmp_path: Path) -> SubprocessRunner:
    return SubprocessRunner(
        runs_dir=str(tmp_path / "runs"),
        timeout_seconds=5,
        python_executable=sys.executable,
    )


@pytest.fixture()
def pipeline(runner: SubprocessRunner) -> EnsemblePipeline:
    debugger = DebuggerAgent(runner=runner, model="test", max_debug_rounds=1)
    return EnsemblePipeline(
        planner=EnsemblePlannerAgent(model="test"),
        ensembler=EnsemblerAgent(debugger=debugger, model="test"),
        runner=runner,
        rounds=2,
    )


class TestEnsemblePipeline:
    """Test Algorithm 3 iterative optimization."""

    def test_selects_best_ensemble_over_rounds(self, pipeline: EnsemblePipeline) -> None:
        plan0 = "average probabilities"
        plan1 = "stack with logistic regression"
        ensembler_model = scripted_ensembler_model({plan0: 0.85, plan1: 0.88})
        with (
            pipeline.planner.agent.override(model=scripted_planner_model(plan0, plan1)),
            pipeline.ensembler.agent.override(model=ensembler_model),
        ):
            result = pipeline.run(_spec(), _solutions(), run_id="ens")
        assert isinstance(result, EnsembleResult)
        assert result.best_score == pytest.approx(0.88)
        assert "0.88" in result.best_code
        assert result.best_artifact.iteration_stage == "ens_optimal"

    def test_never_degrades_below_best_individual(self, pipeline: EnsemblePipeline) -> None:
        plan0 = "average probabilities"
        ensembler_model = scripted_ensembler_model({plan0: 0.75})
        with (
            pipeline.planner.agent.override(
                model=TestModel(custom_output_args=_planner_args("SIMPLE_AVERAGE", plan0))
            ),
            pipeline.ensembler.agent.override(model=ensembler_model),
        ):
            result = pipeline.run(_spec(), _solutions(), run_id="ens2")
        assert result.best_score == pytest.approx(0.82)
        assert "0.82" in result.best_code
        assert result.best_artifact.iteration_stage == "branch_1"

    def test_failed_round_is_logged_and_skipped(self, pipeline: EnsemblePipeline) -> None:
        def broken_model(messages, info):
            return ModelResponse(parts=[TextPart(content="def broken(:\n    pass")])

        with (
            pipeline.planner.agent.override(
                model=TestModel(custom_output_args=_planner_args("SIMPLE_AVERAGE", "avg"))
            ),
            pipeline.ensembler.agent.override(model=FunctionModel(function=broken_model)),
        ):
            result = pipeline.run(_spec(), _solutions(), run_id="ens3")
        assert result.best_score == pytest.approx(0.82)
        assert result.rounds_executed == 2
        records = [
            IterationLogEntry.model_validate_json(line)
            for line in Path(result.logs_path or "").read_text(encoding="utf-8").splitlines()
        ]
        assert all(record.success is False for record in records)
        assert records[0].errors

    def test_iteration_log_streams_round_records(self, pipeline: EnsemblePipeline) -> None:
        plan0 = "average probabilities"
        plan1 = "rank averaging"
        ensembler_model = scripted_ensembler_model({plan0: 0.85, plan1: 0.86})
        with (
            pipeline.planner.agent.override(model=scripted_planner_model(plan0, plan1)),
            pipeline.ensembler.agent.override(model=ensembler_model),
        ):
            result = pipeline.run(_spec(), _solutions(), run_id="ens4")
        records = [
            json.loads(line)
            for line in Path(result.logs_path or "").read_text(encoding="utf-8").splitlines()
        ]
        assert len(records) == 2
        assert records[0]["iteration_id"] == "ens_r0"
        assert records[0]["stage"] == "ENSEMBLING"
        assert records[0]["target_component"] == "ENSEMBLE_SIMPLE_AVERAGE"
        assert records[0]["hypothesis"] == plan0
        assert records[0]["validation_score"] == pytest.approx(0.85)
        assert records[0]["delta_from_baseline"] == pytest.approx(0.03)

    def test_single_candidate_returns_immediately(self, pipeline: EnsemblePipeline) -> None:
        calls = {"count": 0}

        def counting_model(
            messages: list[ModelRequest | ModelResponse], info: AgentInfo
        ) -> ModelResponse:
            calls["count"] += 1
            return ModelResponse(parts=[TextPart(content="print('unexpected call')")])

        solutions = [_artifact("print('Final Validation Performance: 0.80')", 0.80, "branch_0")]
        with (
            pipeline.planner.agent.override(model=FunctionModel(function=counting_model)),
            pipeline.ensembler.agent.override(model=FunctionModel(function=counting_model)),
        ):
            result = pipeline.run(_spec(), solutions, run_id="single")
        assert isinstance(result, EnsembleResult)
        assert result.rounds_executed == 0
        assert result.best_score == pytest.approx(0.80)
        assert "0.80" in result.best_code
        assert result.best_artifact is solutions[0]
        assert result.logs_path is None
        assert calls["count"] == 0

    def test_zero_rounds_returns_best_individual(self, runner: SubprocessRunner) -> None:
        debugger = DebuggerAgent(runner=runner, model="test", max_debug_rounds=1)
        pipeline = EnsemblePipeline(
            planner=EnsemblePlannerAgent(model="test"),
            ensembler=EnsemblerAgent(debugger=debugger, model="test"),
            runner=runner,
            rounds=0,
        )
        calls = {"count": 0}

        def counting_model(
            messages: list[ModelRequest | ModelResponse], info: AgentInfo
        ) -> ModelResponse:
            calls["count"] += 1
            return ModelResponse(parts=[TextPart(content="print('unexpected call')")])

        with (
            pipeline.planner.agent.override(model=FunctionModel(function=counting_model)),
            pipeline.ensembler.agent.override(model=FunctionModel(function=counting_model)),
        ):
            result = pipeline.run(_spec(), _solutions(), run_id="zero")
        assert result.rounds_executed == 0
        assert result.best_score == pytest.approx(0.82)
        assert "0.82" in result.best_code
        assert result.best_artifact.iteration_stage == "branch_1"
        assert result.logs_path is None
        assert calls["count"] == 0

    def test_empty_solutions_raises(self, pipeline: EnsemblePipeline) -> None:
        with pytest.raises(ValueError, match="No candidate solutions"):
            pipeline.run(_spec(), [], run_id="empty")
