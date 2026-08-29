"""Unit tests for the unified execution guardrail pipeline.

Covers ``ExecutionConfig`` controls and the sequential
Leakage -> Usage -> Sandbox -> Debugger orchestration of
``ExecutionGuardrailPipeline.run``.
"""

import sys
from pathlib import Path

import pytest
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.task import ExecutionResult, TaskSpecification
from problem_2_v2.execution.pipeline import ExecutionConfig, ExecutionGuardrailPipeline
from problem_2_v2.guardrails.leakage import DataLeakageCheckerAgent
from problem_2_v2.guardrails.usage import DataUsageCheckerAgent
from problem_2_v2.runner.debugger import DebuggerAgent
from problem_2_v2.runner.sandbox import SubprocessRunner

GOOD_CODE = "print('Final Validation Performance: 0.80')"

BROKEN_CODE = "def broken(:\n    pass\n"
FIXED_CODE = "print('Final Validation Performance: 0.85')"

SUSPICIOUS_BLOCK = "X_test = scaler.transform(test.drop(columns=['label']))"
CORRECTED_BLOCK = "X_test = scaler.fit_transform(test.drop(columns=['label']))"
LEAKY_CODE = (
    "import pandas as pd\n"
    "train = pd.read_csv('./input/train.csv')\n"
    "scaler = StandardScaler()\n"
    f"{SUSPICIOUS_BLOCK}\n"
    "print('Final Validation Performance: 0.75')\n"
)

IMPROVED_CODE = "import numpy as np\n" + GOOD_CODE


def _spec() -> TaskSpecification:
    return TaskSpecification.from_markdown(
        "**Task Type:** TABULAR_CLASSIFICATION\n"
        "**Metric Name:** AUROC\n"
        "**Metric Direction:** MAXIMIZE\n"
        "**Target Variable:** label\n"
        "**Dataset Files:** train.csv\n",
        dataset_dir="/data",
    )


def _leak_clean_args() -> dict[str, object]:
    return {
        "leakage_status": "No Data Leakage",
        "is_leaking": False,
        "suspicious_code_block": None,
        "corrected_code_block": None,
        "explanation": "clean",
    }


def _leak_leaky_args() -> dict[str, object]:
    return {
        "leakage_status": "Yes Data Leakage",
        "is_leaking": True,
        "suspicious_code_block": SUSPICIOUS_BLOCK,
        "corrected_code_block": None,
        "explanation": "leaky",
    }


def _pipeline(tmp_path: Path) -> ExecutionGuardrailPipeline:
    runner = SubprocessRunner(
        runs_dir=str(tmp_path / "runs"),
        timeout_seconds=5,
        python_executable=sys.executable,
    )
    leakage = DataLeakageCheckerAgent(model="test")
    usage = DataUsageCheckerAgent(model="test")
    debugger = DebuggerAgent(runner=runner, model="test", max_debug_rounds=1)
    return ExecutionGuardrailPipeline(
        leakage=leakage,
        usage=usage,
        runner=runner,
        debugger=debugger,
    )


class TestExecutionConfig:
    """Test ExecutionConfig controls."""

    def test_defaults(self) -> None:
        config = ExecutionConfig()
        assert config.timeout_seconds == 600
        assert config.max_debug_rounds == 3
        assert config.sandbox_base_dir == "runs"
        assert config.enable_leakage_check is True
        assert config.enable_usage_check is True
        assert config.production_timeout_seconds == 3600

    def test_overrides(self) -> None:
        config = ExecutionConfig(
            timeout_seconds=30,
            max_debug_rounds=2,
            sandbox_base_dir="sandboxes",
            enable_leakage_check=False,
            enable_usage_check=False,
            production_timeout_seconds=7200,
        )
        assert config.timeout_seconds == 30
        assert config.max_debug_rounds == 2
        assert config.sandbox_base_dir == "sandboxes"
        assert config.enable_leakage_check is False
        assert config.enable_usage_check is False
        assert config.production_timeout_seconds == 7200

    def test_config_drives_default_components(self) -> None:
        pipeline = ExecutionGuardrailPipeline(
            config=ExecutionConfig(timeout_seconds=42, max_debug_rounds=2)
        )
        assert pipeline.runner.timeout_seconds == 42
        assert pipeline.debugger.max_debug_rounds == 2
        assert isinstance(pipeline.leakage, DataLeakageCheckerAgent)
        assert isinstance(pipeline.usage, DataUsageCheckerAgent)


class TestExecutionGuardrailPipeline:
    """Test the sequential Leakage -> Usage -> Sandbox -> Debugger run."""

    def test_successful_execution_returns_parsed_result(self, tmp_path: Path) -> None:
        pipeline = _pipeline(tmp_path)
        with (
            pipeline.leakage.check_agent.override(
                model=TestModel(custom_output_args=_leak_clean_args())
            ),
            pipeline.usage.agent.override(
                model=TestModel(custom_output_text="All the provided information is used.")
            ),
        ):
            result = pipeline.run(GOOD_CODE, _spec(), run_id="r", candidate_id="cand")
        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.validation_score == pytest.approx(0.80)
        assert result.duration_seconds >= 0
        assert pipeline.last_guarded_code == GOOD_CODE
        assert pipeline.last_executed_code == GOOD_CODE

    def test_leakage_pass_repairs_leaky_code(self, tmp_path: Path) -> None:
        pipeline = _pipeline(tmp_path)
        with (
            pipeline.leakage.check_agent.override(
                model=TestModel(custom_output_args=_leak_leaky_args())
            ),
            pipeline.leakage.repair_agent.override(
                model=TestModel(custom_output_text=f"```python\n{CORRECTED_BLOCK}\n```")
            ),
            pipeline.usage.agent.override(
                model=TestModel(custom_output_text="All the provided information is used.")
            ),
        ):
            guarded = pipeline.guard(LEAKY_CODE, _spec())
        assert CORRECTED_BLOCK in guarded
        assert SUSPICIOUS_BLOCK not in guarded

    def test_usage_pass_incorporates_improved_code(self, tmp_path: Path) -> None:
        pipeline = _pipeline(tmp_path)
        with (
            pipeline.leakage.check_agent.override(
                model=TestModel(custom_output_args=_leak_clean_args())
            ),
            pipeline.usage.agent.override(
                model=TestModel(custom_output_text=f"```python\n{IMPROVED_CODE}\n```")
            ),
        ):
            guarded = pipeline.guard(GOOD_CODE, _spec())
        assert "numpy" in guarded

    def test_guardrail_toggles_skip_passes(self, tmp_path: Path) -> None:
        def exploding_model(messages, info):
            raise RuntimeError("guardrail should not be invoked")

        config = ExecutionConfig(enable_leakage_check=False, enable_usage_check=False)
        runner = SubprocessRunner(
            runs_dir=str(tmp_path / "runs"),
            timeout_seconds=5,
            python_executable=sys.executable,
        )
        leakage = DataLeakageCheckerAgent(model="test")
        usage = DataUsageCheckerAgent(model="test")
        debugger = DebuggerAgent(runner=runner, model="test", max_debug_rounds=1)
        pipeline = ExecutionGuardrailPipeline(
            config=config,
            leakage=leakage,
            usage=usage,
            runner=runner,
            debugger=debugger,
        )
        with (
            leakage.check_agent.override(model=FunctionModel(function=exploding_model)),
            leakage.repair_agent.override(model=FunctionModel(function=exploding_model)),
            usage.agent.override(model=FunctionModel(function=exploding_model)),
        ):
            guarded = pipeline.guard(GOOD_CODE, _spec())
        assert guarded == GOOD_CODE

    def test_graceful_degradation_when_guardrail_llm_fails(self, tmp_path: Path) -> None:
        def exploding_model(messages, info):
            raise RuntimeError("LLM backend down")

        pipeline = _pipeline(tmp_path)
        with (
            pipeline.leakage.check_agent.override(model=FunctionModel(function=exploding_model)),
            pipeline.usage.agent.override(
                model=TestModel(custom_output_text="All the provided information is used.")
            ),
        ):
            result = pipeline.run(GOOD_CODE, _spec(), run_id="g", candidate_id="cand")
        assert result.success is True
        assert result.validation_score == pytest.approx(0.80)

    def test_debugger_retries_failing_script(self, tmp_path: Path) -> None:
        pipeline = _pipeline(tmp_path)
        with (
            pipeline.leakage.check_agent.override(
                model=TestModel(custom_output_args=_leak_clean_args())
            ),
            pipeline.usage.agent.override(
                model=TestModel(custom_output_text="All the provided information is used.")
            ),
            pipeline.debugger.agent.override(model=TestModel(custom_output_text=FIXED_CODE)),
        ):
            result = pipeline.run(BROKEN_CODE, _spec(), run_id="dbg", candidate_id="cand")
        assert result.success is True
        assert result.validation_score == pytest.approx(0.85)
        assert pipeline.last_debug_rounds == 1

    def test_failed_execution_returns_failure_result(self, tmp_path: Path) -> None:
        pipeline = _pipeline(tmp_path)
        with (
            pipeline.leakage.check_agent.override(
                model=TestModel(custom_output_args=_leak_clean_args())
            ),
            pipeline.usage.agent.override(
                model=TestModel(custom_output_text="All the provided information is used.")
            ),
            pipeline.debugger.agent.override(model=TestModel(custom_output_text="not python (:")),
        ):
            result = pipeline.run(BROKEN_CODE, _spec(), run_id="fail", candidate_id="cand")
        assert result.success is False
        assert result.validation_score is None
        assert result.returncode != 0

    def test_execution_uses_isolated_candidate_sandbox(self, tmp_path: Path) -> None:
        pipeline = _pipeline(tmp_path)
        with (
            pipeline.leakage.check_agent.override(
                model=TestModel(custom_output_args=_leak_clean_args())
            ),
            pipeline.usage.agent.override(
                model=TestModel(custom_output_text="All the provided information is used.")
            ),
        ):
            pipeline.run(GOOD_CODE, _spec(), run_id="iso", candidate_id="candA")
            pipeline.run(GOOD_CODE, _spec(), run_id="iso", candidate_id="candB")
        runs = tmp_path / "runs" / "iso"
        assert (runs / "sandbox_candA" / "solution.py").exists()
        assert (runs / "sandbox_candB" / "solution.py").exists()
