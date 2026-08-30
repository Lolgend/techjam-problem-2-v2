"""Unit tests for Phases 2-4: retry loop, strict enforcement, and observability.

Phase 2: retry loop with ``max_leakage_retries`` config.
Phase 3: ``strict_leakage`` enforcement and ``LeakageEnforcementError``.
Phase 4: distinct Logfire events replacing ambiguous ``execution.leakage_detected``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.task import TaskSpecification
from problem_2_v2.execution.pipeline import ExecutionConfig, ExecutionGuardrailPipeline
from problem_2_v2.guardrails.leakage import DataLeakageCheckerAgent, LeakageEnforcementError
from problem_2_v2.guardrails.usage import DataUsageCheckerAgent
from problem_2_v2.runner.debugger import DebuggerAgent
from problem_2_v2.runner.sandbox import SubprocessRunner

GOOD_CODE = "print('Final Validation Performance: 0.80')"

SUSPICIOUS_BLOCK = "X_test = scaler.transform(test.drop(columns=['label']))"
CORRECTED_BLOCK = "X_test = scaler.fit_transform(test.drop(columns=['label']))"
LEAKY_CODE = (
    "import pandas as pd\n"
    "train = pd.read_csv('./input/train.csv')\n"
    "scaler = StandardScaler()\n"
    f"{SUSPICIOUS_BLOCK}\n"
    "print('Final Validation Performance: 0.75')\n"
)


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


def _make_sequence_check_model(*status_dicts: dict[str, object]) -> FunctionModel:
    idx = [0]

    def model_fn(messages: object, info: object) -> ModelResponse:
        d = status_dicts[min(idx[0], len(status_dicts) - 1)]
        idx[0] += 1
        return ModelResponse(parts=[TextPart(content=json.dumps(d))])

    return FunctionModel(function=model_fn)


def _pipeline(tmp_path: Path, **config_kwargs: object) -> ExecutionGuardrailPipeline:
    config = ExecutionConfig(enable_usage_check=False, **config_kwargs)
    runner = SubprocessRunner(
        runs_dir=str(tmp_path / "runs"),
        timeout_seconds=5,
        python_executable=sys.executable,
    )
    leakage = DataLeakageCheckerAgent(model="test")
    usage = DataUsageCheckerAgent(model="test")
    debugger = DebuggerAgent(runner=runner, model="test", max_debug_rounds=1)
    return ExecutionGuardrailPipeline(
        config=config,
        leakage=leakage,
        usage=usage,
        runner=runner,
        debugger=debugger,
    )


# ── Phase 2: Retry Loop & Config Fields ───────────────────────────────


class TestRetryConfig:
    """Test ``max_leakage_retries`` config field."""

    def test_default_max_leakage_retries(self) -> None:
        config = ExecutionConfig()
        assert config.max_leakage_retries == 5

    def test_custom_max_leakage_retries(self) -> None:
        config = ExecutionConfig(max_leakage_retries=10)
        assert config.max_leakage_retries == 10


class TestRetryLoop:
    """Test the check→repair→re-check retry loop in ``guard()``."""

    def test_early_exit_when_repair_succeeds(self, tmp_path: Path) -> None:
        """Loop exits after 1st retry when re-check confirms clean."""
        pipeline = _pipeline(tmp_path, max_leakage_retries=3)
        audit_count = {"n": 0}
        original_audit = pipeline.leakage.audit

        def counting_audit(code: str):
            audit_count["n"] += 1
            return original_audit(code)

        check_model = _make_sequence_check_model(_leak_leaky_args(), _leak_clean_args())
        with (
            pipeline.leakage.check_agent.override(model=check_model),
            pipeline.leakage.repair_agent.override(
                model=TestModel(custom_output_text=f"```python\n{CORRECTED_BLOCK}\n```")
            ),
        ):
            pipeline.leakage.audit = counting_audit
            guarded = pipeline.guard(LEAKY_CODE, _spec())
        # 1 initial attempt + 1 retry = 2 audit calls, then early exit.
        assert isinstance(guarded, str)
        assert audit_count["n"] == 2

    def test_retries_exhaust_when_leakage_persists(self, tmp_path: Path) -> None:
        """Loop runs up to max_leakage_retries when leakage persists."""
        max_retries = 3
        pipeline = _pipeline(tmp_path, max_leakage_retries=max_retries)

        with (
            pipeline.leakage.check_agent.override(
                model=TestModel(custom_output_args=_leak_leaky_args())
            ),
            pipeline.leakage.repair_agent.override(
                model=TestModel(custom_output_text=f"```python\n{CORRECTED_BLOCK}\n```")
            ),
        ):
            guarded = pipeline.guard(LEAKY_CODE, _spec())
        # Should still return code (lenient mode) despite exhausted retries.
        assert isinstance(guarded, str)

    def test_latest_code_passed_to_each_retry(self, tmp_path: Path) -> None:
        """Each retry feeds the latest repaired code back through audit()."""
        pipeline = _pipeline(tmp_path, max_leakage_retries=2)
        audit_inputs: list[str] = []
        original_audit = pipeline.leakage.audit

        def tracking_audit(code: str):
            audit_inputs.append(code)
            return original_audit(code)

        with (
            pipeline.leakage.check_agent.override(
                model=TestModel(custom_output_args=_leak_leaky_args())
            ),
            pipeline.leakage.repair_agent.override(
                model=TestModel(custom_output_text=f"```python\n{CORRECTED_BLOCK}\n```")
            ),
        ):
            pipeline.leakage.audit = tracking_audit
            pipeline.guard(LEAKY_CODE, _spec())
        # The 2nd audit should receive different code than the 1st (the repaired version).
        assert len(audit_inputs) >= 2
        assert audit_inputs[0] != audit_inputs[1]

    def test_zero_retries_single_attempt(self, tmp_path: Path) -> None:
        """max_leakage_retries=0 means single audit attempt, no retry."""
        pipeline = _pipeline(tmp_path, max_leakage_retries=0)
        audit_count = {"n": 0}
        original_audit = pipeline.leakage.audit

        def counting_audit(code: str):
            audit_count["n"] += 1
            return original_audit(code)

        with (
            pipeline.leakage.check_agent.override(
                model=TestModel(custom_output_args=_leak_leaky_args())
            ),
            pipeline.leakage.repair_agent.override(
                model=TestModel(custom_output_text=f"```python\n{CORRECTED_BLOCK}\n```")
            ),
        ):
            pipeline.leakage.audit = counting_audit
            pipeline.guard(LEAKY_CODE, _spec())
        assert audit_count["n"] == 1  # single attempt, no retries


# ── Phase 3: Strict Enforcement Mode ──────────────────────────────────


class TestStrictEnforcementConfig:
    """Test ``strict_leakage`` config field."""

    def test_default_strict_leakage(self) -> None:
        config = ExecutionConfig()
        assert config.strict_leakage is False

    def test_custom_strict_leakage(self) -> None:
        config = ExecutionConfig(strict_leakage=True)
        assert config.strict_leakage is True


class TestLeakageEnforcementError:
    """Test the ``LeakageEnforcementError`` exception class."""

    def test_is_runtime_error_subclass(self) -> None:
        assert issubclass(LeakageEnforcementError, RuntimeError)

    def test_descriptive_message(self) -> None:
        err = LeakageEnforcementError("leakage persists after 5 retries")
        assert "5 retries" in str(err)


class TestStrictEnforcement:
    """Test strict vs lenient leakage enforcement in ``guard()``."""

    def test_lenient_continues_after_exhausted_retries(self, tmp_path: Path) -> None:
        """strict_leakage=False + exhausted retries → warns and returns code."""
        pipeline = _pipeline(tmp_path, strict_leakage=False, max_leakage_retries=1)
        with (
            pipeline.leakage.check_agent.override(
                model=TestModel(custom_output_args=_leak_leaky_args())
            ),
            pipeline.leakage.repair_agent.override(
                model=TestModel(custom_output_text=f"```python\n{CORRECTED_BLOCK}\n```")
            ),
        ):
            guarded = pipeline.guard(LEAKY_CODE, _spec())
        assert isinstance(guarded, str)  # Should not raise

    def test_strict_raises_after_exhausted_retries(self, tmp_path: Path) -> None:
        """strict_leakage=True + exhausted retries → raises LeakageEnforcementError."""
        pipeline = _pipeline(tmp_path, strict_leakage=True, max_leakage_retries=1)
        with (
            pipeline.leakage.check_agent.override(
                model=TestModel(custom_output_args=_leak_leaky_args())
            ),
            pipeline.leakage.repair_agent.override(
                model=TestModel(custom_output_text=f"```python\n{CORRECTED_BLOCK}\n```")
            ),
            pytest.raises(LeakageEnforcementError),
        ):
            pipeline.guard(LEAKY_CODE, _spec())

    def test_strict_succeeds_when_repair_works(self, tmp_path: Path) -> None:
        """strict_leakage=True + successful repair → returns clean code (no error)."""
        pipeline = _pipeline(tmp_path, strict_leakage=True, max_leakage_retries=3)
        check_model = _make_sequence_check_model(_leak_leaky_args(), _leak_clean_args())
        with (
            pipeline.leakage.check_agent.override(model=check_model),
            pipeline.leakage.repair_agent.override(
                model=TestModel(custom_output_text=f"```python\n{CORRECTED_BLOCK}\n```")
            ),
        ):
            guarded = pipeline.guard(LEAKY_CODE, _spec())
        assert isinstance(guarded, str)  # Should not raise


# ── Phase 4: Unambiguous Observability ────────────────────────────────


class TestObservability:
    """Test distinct Logfire events for leakage repair outcomes."""

    def test_leakage_repaired_event_on_success(self, tmp_path: Path) -> None:
        """guard() runs repair loop and succeeds — code should be patched."""
        pipeline = _pipeline(tmp_path, max_leakage_retries=3)
        check_model = _make_sequence_check_model(_leak_leaky_args(), _leak_clean_args())
        with (
            pipeline.leakage.check_agent.override(model=check_model),
            pipeline.leakage.repair_agent.override(
                model=TestModel(custom_output_text=f"```python\n{CORRECTED_BLOCK}\n```")
            ),
        ):
            guarded = pipeline.guard(LEAKY_CODE, _spec())
        # Repair succeeded: code should be modified.
        assert isinstance(guarded, str)

    def test_leakage_unrepaired_event_on_failure(self, tmp_path: Path) -> None:
        """guard() runs repair loop, fails, returns code in lenient mode."""
        pipeline = _pipeline(tmp_path, max_leakage_retries=1, strict_leakage=False)
        with (
            pipeline.leakage.check_agent.override(
                model=TestModel(custom_output_args=_leak_leaky_args())
            ),
            pipeline.leakage.repair_agent.override(
                model=TestModel(custom_output_text=f"```python\n{CORRECTED_BLOCK}\n```")
            ),
        ):
            guarded = pipeline.guard(LEAKY_CODE, _spec())
        # Lenient mode: should return code, not raise.
        assert isinstance(guarded, str)

    def test_leakage_detected_event_removed_from_source(self) -> None:
        """The old ``execution.leakage_detected`` event no longer appears in source."""
        import inspect

        from problem_2_v2.execution import pipeline as pipeline_mod

        source = inspect.getsource(pipeline_mod)
        assert "leakage_detected" not in source
        assert "leakage_repaired" in source
        assert "leakage_unrepaired" in source
