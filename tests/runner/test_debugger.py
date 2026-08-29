"""Unit tests for the autonomous debugging agent."""

import sys
from pathlib import Path

import pytest
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from problem_2_v2.runner.debugger import DebuggerAgent, DebugOutcome
from problem_2_v2.runner.sandbox import SubprocessRunner

BROKEN_SYNTAX = "def f(:\n    return 1\n"
BROKEN_RUNTIME = "raise ValueError('boom')\n"
FIXED_CODE = "print('Final Validation Performance: 0.5')"
STILL_BROKEN = "def f(:\n    return 2\n"


@pytest.fixture()
def runner(tmp_path: Path) -> SubprocessRunner:
    return SubprocessRunner(
        runs_dir=str(tmp_path / "runs"),
        timeout_seconds=5,
        python_executable=sys.executable,
    )


@pytest.fixture()
def debugger(runner: SubprocessRunner) -> DebuggerAgent:
    return DebuggerAgent(runner=runner, model="test", max_debug_rounds=3)


class TestDebuggerAgent:
    """Test the iterative repair loop."""

    def test_recovers_syntax_error_in_one_round(self, debugger: DebuggerAgent) -> None:
        with debugger.agent.override(model=TestModel(custom_output_text=FIXED_CODE)):
            outcome = debugger.debug(BROKEN_SYNTAX)
        assert isinstance(outcome, DebugOutcome)
        assert outcome.recovered is True
        assert outcome.debug_rounds == 1
        assert outcome.result.success is True
        assert outcome.result.validation_score == pytest.approx(0.5)

    def test_repair_strips_markdown_fences(self, debugger: DebuggerAgent) -> None:
        fenced_fix = f"```python\n{FIXED_CODE}\n```"
        with debugger.agent.override(model=TestModel(custom_output_text=fenced_fix)):
            outcome = debugger.debug(BROKEN_SYNTAX)
        assert outcome.recovered is True
        assert outcome.debug_rounds == 1
        assert "```" not in outcome.code
        assert outcome.result.validation_score == pytest.approx(0.5)

    def test_recovers_runtime_error(self, debugger: DebuggerAgent) -> None:
        with debugger.agent.override(model=TestModel(custom_output_text=FIXED_CODE)):
            outcome = debugger.debug(BROKEN_RUNTIME)
        assert outcome.recovered is True
        assert outcome.debug_rounds == 1

    def test_gives_up_after_max_rounds(self, debugger: DebuggerAgent) -> None:
        with debugger.agent.override(model=TestModel(custom_output_text=STILL_BROKEN)):
            outcome = debugger.debug(BROKEN_SYNTAX)
        assert outcome.recovered is False
        assert outcome.debug_rounds == 3

    def test_successful_code_skips_debug_rounds(self, debugger: DebuggerAgent) -> None:
        outcome = debugger.debug(FIXED_CODE)
        assert outcome.recovered is True
        assert outcome.debug_rounds == 0

    def test_traceback_is_passed_to_the_model(self, runner: SubprocessRunner) -> None:
        captured: dict[str, str] = {}

        def capturing_model(messages, info):
            prompt_text = messages[-1].parts[0].content
            captured["prompt"] = prompt_text
            return ModelResponse(parts=[TextPart(content=FIXED_CODE)])

        debugger = DebuggerAgent(runner=runner, model="test", max_debug_rounds=2)
        with debugger.agent.override(model=FunctionModel(function=capturing_model)):
            outcome = debugger.debug(BROKEN_RUNTIME)
        assert outcome.recovered is True
        assert "boom" in captured["prompt"]
        assert "Code with an error" in captured["prompt"] or "Error" in captured["prompt"]

    def test_missing_score_line_triggers_debugging(self, debugger: DebuggerAgent) -> None:
        no_score = "print('hello')"
        with debugger.agent.override(model=TestModel(custom_output_text=FIXED_CODE)):
            outcome = debugger.debug(no_score)
        assert outcome.recovered is True
        assert outcome.debug_rounds == 1

    def test_llm_failure_does_not_crash_the_loop(self, runner: SubprocessRunner) -> None:
        def exploding_model(messages, info):
            raise RuntimeError("LLM backend down")

        debugger = DebuggerAgent(runner=runner, model="test", max_debug_rounds=2)
        with debugger.agent.override(model=FunctionModel(function=exploding_model)):
            outcome = debugger.debug(BROKEN_SYNTAX)
        assert outcome.recovered is False
        assert outcome.debug_rounds == 2
