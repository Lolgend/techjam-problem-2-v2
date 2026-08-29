"""Unit tests for the nested refinement pipeline orchestrator."""

import json
import sys
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.task import TaskSpecification
from problem_2_v2.guardrails.leakage import DataLeakageCheckerAgent
from problem_2_v2.guardrails.usage import DataUsageCheckerAgent
from problem_2_v2.refinement.ablation import AblationAgent, AblationSummarizerAgent
from problem_2_v2.refinement.coder import CoderAgent
from problem_2_v2.refinement.extractor import CodeBlockExtractorAgent
from problem_2_v2.refinement.pipeline import IterationLogRecord, RefinementPipeline
from problem_2_v2.refinement.planner import RefinementPlannerAgent
from problem_2_v2.runner.debugger import DebuggerAgent
from problem_2_v2.runner.sandbox import SubprocessRunner

TARGET_BLOCK = "model = 'baseline'\nprint('Final Validation Performance: 0.80')"

INITIAL_CODE = "x = 1\nmodel = 'baseline'\nprint('Final Validation Performance: 0.80')\n"

REFINED_BLOCK = "model = 'boosted'\nprint('Final Validation Performance: 0.85')"

IMPROVED_SCRIPT = "x = 1\nmodel = 'boosted'\nprint('Final Validation Performance: 0.85')\n"


def _spec() -> TaskSpecification:
    return TaskSpecification.from_markdown(
        "**Task Name:** Demo\n"
        "**Task Type:** TABULAR_CLASSIFICATION\n"
        "**Metric Name:** AUROC\n"
        "**Metric Direction:** MAXIMIZE\n"
        "**Target Variable:** label\n"
        "**Dataset Files:** train.csv\n"
        "**Description:** classify.\n",
        dataset_dir="/data",
    )


def _report_args() -> dict[str, object]:
    return {
        "baseline_score": 0.80,
        "ablation_results": [
            {
                "variant_id": "model",
                "validation_score": 0.82,
                "delta_from_baseline": 0.02,
                "summary": "Model architecture mattered most.",
            }
        ],
        "highest_impact_component": "model",
        "raw_log_summary": "...",
    }


def _leak_clean_args() -> dict[str, object]:
    return {
        "leakage_status": "No Data Leakage",
        "is_leaking": False,
        "suspicious_code_block": None,
        "corrected_code_block": None,
        "explanation": "clean",
    }


@pytest.fixture()
def runner(tmp_path: Path) -> SubprocessRunner:
    return SubprocessRunner(
        runs_dir=str(tmp_path / "runs"),
        timeout_seconds=5,
        python_executable=sys.executable,
    )


@pytest.fixture()
def pipeline(runner: SubprocessRunner) -> RefinementPipeline:
    debugger = DebuggerAgent(runner=runner, model="test", max_debug_rounds=1)
    return RefinementPipeline(
        ablation=AblationAgent(model="test"),
        summarizer=AblationSummarizerAgent(runner=runner, model="test"),
        extractor=CodeBlockExtractorAgent(model="test"),
        planner=RefinementPlannerAgent(model="test"),
        coder=CoderAgent(model="test"),
        leakage=DataLeakageCheckerAgent(model="test"),
        usage=DataUsageCheckerAgent(model="test"),
        debugger=debugger,
        runner=runner,
        outer_loops=1,
        inner_loops=2,
    )


def _extractor_args() -> list[dict[str, object]]:
    return [
        {
            "code_block": TARGET_BLOCK,
            "plan": "Replace the linear model with gradient boosted trees.",
            "category": "MODEL_ARCHITECTURE",
        }
    ]


def _run_pipeline(pipeline: RefinementPipeline, coder_output: str, run_id: str = "r"):
    """Run the pipeline with all non-coder agents scripted."""
    with (
        pipeline.ablation.agent.override(
            model=TestModel(custom_output_text="print('ablation done')")
        ),
        pipeline.summarizer.agent.override(model=TestModel(custom_output_args=_report_args())),
        pipeline.extractor.agent.override(model=TestModel(custom_output_args=_extractor_args())),
        pipeline.planner.agent.override(model=TestModel(custom_output_text="improve the model")),
        pipeline.coder.agent.override(
            model=TestModel(custom_output_text=f"```python\n{coder_output}\n```")
        ),
        pipeline.leakage.check_agent.override(
            model=TestModel(custom_output_args=_leak_clean_args())
        ),
        pipeline.usage.agent.override(
            model=TestModel(custom_output_text="All the provided information is used.")
        ),
    ):
        return pipeline.refine(_spec(), INITIAL_CODE, initial_score=0.80, run_id=run_id)


class TestRefinementPipeline:
    """Test the nested outer x inner loop orchestrator."""

    def test_run_produces_improved_solution(self, pipeline: RefinementPipeline) -> None:
        result = _run_pipeline(pipeline, REFINED_BLOCK)
        assert result.final_score == pytest.approx(0.85)
        assert "boosted" in result.final_code
        assert result.final_code != INITIAL_CODE
        assert result.final_code == IMPROVED_SCRIPT

    def test_iteration_log_is_streamed_to_jsonl(
        self, pipeline: RefinementPipeline, tmp_path: Path
    ) -> None:
        result = _run_pipeline(pipeline, REFINED_BLOCK, run_id="logtest")
        logs_path = Path(result.logs_path or "")
        assert logs_path.exists()
        records = [json.loads(line) for line in logs_path.read_text(encoding="utf-8").splitlines()]
        assert len(records) == pipeline.outer_loops * pipeline.inner_loops
        first = records[0]
        assert first["outer_iteration"] == 0
        assert first["inner_iteration"] == 0
        assert "plan" in first
        assert first["success"] is True

    def test_regression_is_recorded_but_final_keeps_best(
        self, pipeline: RefinementPipeline
    ) -> None:
        regressing = "model = 'weak'\nprint('Final Validation Performance: 0.79')"
        result = _run_pipeline(pipeline, regressing)
        assert result.final_score == pytest.approx(0.80)
        assert result.final_code == INITIAL_CODE

    def test_invalid_patch_does_not_crash_pipeline(
        self, pipeline: RefinementPipeline, tmp_path: Path
    ) -> None:
        broken = "if True:"
        result = _run_pipeline(pipeline, broken, run_id="broken")
        assert result.final_score == pytest.approx(0.80)
        logs_path = Path(result.logs_path or "")
        records = [json.loads(line) for line in logs_path.read_text(encoding="utf-8").splitlines()]
        assert all(record["success"] is False for record in records)

    def test_log_records_carry_score_and_delta(
        self, pipeline: RefinementPipeline, tmp_path: Path
    ) -> None:
        result = _run_pipeline(pipeline, REFINED_BLOCK, run_id="scored")
        logs_path = Path(result.logs_path or "")
        records = [
            IterationLogRecord.model_validate_json(line)
            for line in logs_path.read_text(encoding="utf-8").splitlines()
        ]
        assert records[0].validation_score == pytest.approx(0.85)
        assert records[0].delta_from_baseline == pytest.approx(0.05)

    def test_planner_failure_does_not_crash_run(self, pipeline: RefinementPipeline) -> None:
        def exploding_model(messages, info):
            raise RuntimeError("LLM backend down")

        from pydantic_ai.models.function import FunctionModel

        with (
            pipeline.ablation.agent.override(
                model=TestModel(custom_output_text="print('ablation done')")
            ),
            pipeline.summarizer.agent.override(model=TestModel(custom_output_args=_report_args())),
            pipeline.extractor.agent.override(
                model=TestModel(custom_output_args=_extractor_args())
            ),
            pipeline.planner.agent.override(model=FunctionModel(function=exploding_model)),
            pipeline.coder.agent.override(
                model=TestModel(custom_output_text=f"```python\n{REFINED_BLOCK}\n```")
            ),
            pipeline.leakage.check_agent.override(
                model=TestModel(custom_output_args=_leak_clean_args())
            ),
            pipeline.usage.agent.override(
                model=TestModel(custom_output_text="All the provided information is used.")
            ),
        ):
            result = pipeline.refine(
                _spec(), INITIAL_CODE, initial_score=0.80, run_id="plannerfail"
            )
        assert result.final_score == pytest.approx(0.85)
        assert result.final_code == IMPROVED_SCRIPT

    def test_extractor_failure_does_not_crash_run(self, pipeline: RefinementPipeline) -> None:
        def exploding_model(messages, info):
            raise RuntimeError("LLM backend down")

        from pydantic_ai.models.function import FunctionModel

        with (
            pipeline.ablation.agent.override(
                model=TestModel(custom_output_text="print('ablation done')")
            ),
            pipeline.summarizer.agent.override(model=TestModel(custom_output_args=_report_args())),
            pipeline.extractor.agent.override(model=FunctionModel(function=exploding_model)),
            pipeline.leakage.check_agent.override(
                model=TestModel(custom_output_args=_leak_clean_args())
            ),
            pipeline.usage.agent.override(
                model=TestModel(custom_output_text="All the provided information is used.")
            ),
        ):
            result = pipeline.refine(
                _spec(), INITIAL_CODE, initial_score=0.80, run_id="extractorfail"
            )
        assert result.final_score == pytest.approx(0.80)
        assert result.final_code == INITIAL_CODE
