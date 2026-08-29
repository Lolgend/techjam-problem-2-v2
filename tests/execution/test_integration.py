"""Integration tests: pipelines delegate execution to ExecutionGuardrailPipeline.

Verifies that ``RefinementPipeline`` and ``EnsemblePipeline`` route script
execution through the unified ``ExecutionGuardrailPipeline`` instead of
calling individual guardrails and runners directly.
"""

import sys
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.task import ExecutionResult, PipelineArtifact, TaskSpecification
from problem_2_v2.ensembling.ensembler import EnsemblerAgent
from problem_2_v2.ensembling.pipeline import EnsemblePipeline
from problem_2_v2.ensembling.planner import EnsemblePlannerAgent
from problem_2_v2.execution.pipeline import ExecutionGuardrailPipeline
from problem_2_v2.guardrails.leakage import DataLeakageCheckerAgent
from problem_2_v2.guardrails.usage import DataUsageCheckerAgent
from problem_2_v2.refinement.ablation import AblationAgent, AblationSummarizerAgent
from problem_2_v2.refinement.coder import CoderAgent
from problem_2_v2.refinement.extractor import CodeBlockExtractorAgent
from problem_2_v2.refinement.pipeline import RefinementPipeline
from problem_2_v2.refinement.planner import RefinementPlannerAgent
from problem_2_v2.runner.debugger import DebuggerAgent
from problem_2_v2.runner.sandbox import SubprocessRunner

TARGET_BLOCK = "model = 'baseline'\nprint('Final Validation Performance: 0.80')"
INITIAL_CODE = "x = 1\nmodel = 'baseline'\nprint('Final Validation Performance: 0.80')\n"
REFINED_BLOCK = "model = 'boosted'\nprint('Final Validation Performance: 0.85')"
ENSEMBLE_CODE = "print('Final Validation Performance: 0.90')"


class SpyExecutionPipeline(ExecutionGuardrailPipeline):
    """Records every run() call to prove pipeline delegation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.run_calls: list[tuple[str, str]] = []

    def run(
        self,
        code: str,
        spec: TaskSpecification,
        run_id: str = "exec",
        candidate_id: str = "candidate",
    ) -> ExecutionResult:
        self.run_calls.append((run_id, candidate_id))
        return super().run(code, spec, run_id=run_id, candidate_id=candidate_id)


def _spec() -> TaskSpecification:
    return TaskSpecification.from_markdown(
        "**Task Type:** TABULAR_CLASSIFICATION\n"
        "**Metric Name:** AUROC\n"
        "**Metric Direction:** MAXIMIZE\n"
        "**Target Variable:** label\n"
        "**Dataset Files:** train.csv\n",
        dataset_dir="/data",
    )


def _execution_pipeline(tmp_path: Path) -> SpyExecutionPipeline:
    runner = SubprocessRunner(
        runs_dir=str(tmp_path / "runs"),
        timeout_seconds=5,
        python_executable=sys.executable,
    )
    return SpyExecutionPipeline(
        leakage=DataLeakageCheckerAgent(model="test"),
        usage=DataUsageCheckerAgent(model="test"),
        runner=runner,
        debugger=DebuggerAgent(runner=runner, model="test", max_debug_rounds=1),
    )


def _leak_clean_args() -> dict[str, object]:
    return {
        "leakage_status": "No Data Leakage",
        "is_leaking": False,
        "suspicious_code_block": None,
        "corrected_code_block": None,
        "explanation": "clean",
    }


def _report_args() -> dict[str, object]:
    return {
        "baseline_score": 0.80,
        "ablation_results": [
            {
                "variant_id": "model",
                "validation_score": 0.82,
                "delta_from_baseline": 0.02,
                "summary": "Model mattered most.",
            }
        ],
        "highest_impact_component": "model",
        "raw_log_summary": "...",
    }


def _extractor_args() -> list[dict[str, object]]:
    return [
        {
            "code_block": TARGET_BLOCK,
            "plan": "Replace the linear model with gradient boosted trees.",
            "category": "MODEL_ARCHITECTURE",
        }
    ]


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


class TestPipelineIntegration:
    """Verify both downstream pipelines delegate to ExecutionGuardrailPipeline."""

    def test_refinement_pipeline_delegates_to_execution_pipeline(self, tmp_path: Path) -> None:
        exec_pl = _execution_pipeline(tmp_path)
        runner = exec_pl.runner
        pipeline = RefinementPipeline(
            ablation=AblationAgent(model="test"),
            summarizer=AblationSummarizerAgent(runner=runner, model="test"),
            extractor=CodeBlockExtractorAgent(model="test"),
            planner=RefinementPlannerAgent(model="test"),
            coder=CoderAgent(model="test"),
            execution=exec_pl,
            outer_loops=1,
            inner_loops=2,
        )
        with (
            pipeline.ablation.agent.override(
                model=TestModel(custom_output_text="print('ablation done')")
            ),
            pipeline.summarizer.agent.override(model=TestModel(custom_output_args=_report_args())),
            pipeline.extractor.agent.override(
                model=TestModel(custom_output_args=_extractor_args())
            ),
            pipeline.planner.agent.override(
                model=TestModel(custom_output_text="improve the model")
            ),
            pipeline.coder.agent.override(
                model=TestModel(custom_output_text=f"```python\n{REFINED_BLOCK}\n```")
            ),
            exec_pl.leakage.check_agent.override(
                model=TestModel(custom_output_args=_leak_clean_args())
            ),
            exec_pl.usage.agent.override(
                model=TestModel(custom_output_text="All the provided information is used.")
            ),
        ):
            result = pipeline.refine(_spec(), INITIAL_CODE, initial_score=0.80, run_id="intr")
        assert exec_pl.run_calls == [("intr", "refine_t0_k0"), ("intr", "refine_t0_k1")]
        assert result.final_score == pytest.approx(0.85)

    def test_ensemble_pipeline_delegates_to_execution_pipeline(self, tmp_path: Path) -> None:
        exec_pl = _execution_pipeline(tmp_path)
        runner = exec_pl.runner
        ensembler = EnsemblerAgent(
            debugger=DebuggerAgent(runner=runner, model="test", max_debug_rounds=1),
            model="test",
        )
        pipeline = EnsemblePipeline(
            planner=EnsemblePlannerAgent(model="test"),
            ensembler=ensembler,
            runner=runner,
            rounds=2,
            execution=exec_pl,
        )
        with (
            pipeline.planner.agent.override(
                model=TestModel(
                    custom_output_args={
                        "method": "SIMPLE_AVERAGE",
                        "natural_language_plan": "average the probabilities",
                        "meta_learner_type": None,
                    }
                )
            ),
            pipeline.ensembler.agent.override(
                model=TestModel(custom_output_text=f"```python\n{ENSEMBLE_CODE}\n```")
            ),
            exec_pl.leakage.check_agent.override(
                model=TestModel(custom_output_args=_leak_clean_args())
            ),
            exec_pl.usage.agent.override(
                model=TestModel(custom_output_text="All the provided information is used.")
            ),
        ):
            result = pipeline.run(_spec(), _solutions(), run_id="inte")
        assert ensembler.execution is exec_pl
        assert exec_pl.run_calls == [("inte", "ens_r0"), ("inte", "ens_r1")]
        assert result.best_score == pytest.approx(0.90)
