"""Unit tests for the code ensembler agent."""

import sys
from pathlib import Path

import pytest
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.enums import EnsembleMethod
from problem_2_v2.contracts.guardrails import EnsembleStrategy
from problem_2_v2.contracts.task import PipelineArtifact, TaskSpecification
from problem_2_v2.ensembling.ensembler import EnsemblerAgent, EnsembleRun
from problem_2_v2.runner.debugger import DebuggerAgent
from problem_2_v2.runner.sandbox import SubprocessRunner

SOLUTION_1 = "pred_a = model_a.predict_proba(X)"
SOLUTION_2 = "pred_b = model_b.predict_proba(X)"

ENSEMBLE_CODE = "print('Final Validation Performance: 0.85')"

SUBMISSION_CODE = (
    "from pathlib import Path\n"
    "Path('./final').mkdir(parents=True, exist_ok=True)\n"
    "Path('./final/submission.csv').write_text('id,pred\\n1,0.5\\n')\n"
    "print('Final Validation Performance: 0.86')\n"
)

BROKEN_CODE = "def broken(:\n    pass\n"
FIXED_CODE = "print('Final Validation Performance: 0.87')"


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


def _strategy() -> EnsembleStrategy:
    return EnsembleStrategy(
        method=EnsembleMethod.SIMPLE_AVERAGE,
        natural_language_plan="average the predicted probabilities",
        meta_learner_type=None,
        candidate_solution_ids=["branch_0", "branch_1"],
        code_template=None,
    )


@pytest.fixture()
def runner(tmp_path: Path) -> SubprocessRunner:
    return SubprocessRunner(
        runs_dir=str(tmp_path / "runs"),
        timeout_seconds=5,
        python_executable=sys.executable,
    )


@pytest.fixture()
def ensembler(runner: SubprocessRunner) -> EnsemblerAgent:
    debugger = DebuggerAgent(runner=runner, model="test", max_debug_rounds=1)
    return EnsemblerAgent(debugger=debugger, model="test")


class TestEnsemblerAgent:
    """Test Figure 18 single-file ensemble synthesis and execution."""

    def test_ensembles_and_scores(self, ensembler: EnsemblerAgent) -> None:
        solutions = [
            _artifact(SOLUTION_1, 0.80, "branch_0"),
            _artifact(SOLUTION_2, 0.82, "branch_1"),
        ]
        with ensembler.agent.override(
            model=TestModel(custom_output_text=f"```python\n{ENSEMBLE_CODE}\n```")
        ):
            run = ensembler.ensemble(_spec(), solutions, _strategy(), run_id="r", round_index=0)
        assert isinstance(run, EnsembleRun)
        assert run.success is True
        assert run.score == pytest.approx(0.85)
        assert "```" not in run.code

    def test_produces_submission_file(self, ensembler: EnsemblerAgent) -> None:
        solutions = [_artifact(SOLUTION_1, 0.80, "branch_0")]
        with ensembler.agent.override(
            model=TestModel(custom_output_text=f"```python\n{SUBMISSION_CODE}\n```")
        ):
            run = ensembler.ensemble(_spec(), solutions, _strategy(), run_id="r2", round_index=1)
        assert run.submission_path is not None
        assert Path(run.submission_path).exists()

    def test_marks_invalid_code_as_failed(self, ensembler: EnsemblerAgent) -> None:
        solutions = [_artifact(SOLUTION_1, 0.80, "branch_0")]
        with ensembler.agent.override(
            model=TestModel(custom_output_text="```python\nnot python (:\n```")
        ):
            run = ensembler.ensemble(_spec(), solutions, _strategy(), run_id="r3", round_index=0)
        assert run.success is False
        assert run.score is None

    def test_recovers_broken_script_via_debugger(self, ensembler: EnsemblerAgent) -> None:
        solutions = [_artifact(SOLUTION_1, 0.80, "branch_0")]
        with (
            ensembler.agent.override(
                model=TestModel(custom_output_text=f"```python\n{BROKEN_CODE}\n```")
            ),
            ensembler.debugger.agent.override(model=TestModel(custom_output_text=FIXED_CODE)),
        ):
            run = ensembler.ensemble(_spec(), solutions, _strategy(), run_id="r4", round_index=0)
        assert run.success is True
        assert run.score == pytest.approx(0.87)
        assert run.debug_rounds == 1

    def test_prompt_contains_solutions_and_plan(self, ensembler: EnsemblerAgent) -> None:
        captured: dict[str, str] = {}

        def capturing_model(messages, info):
            captured["prompt"] = messages[-1].parts[0].content
            return ModelResponse(parts=[TextPart(content=ENSEMBLE_CODE)])

        solutions = [_artifact(SOLUTION_1, 0.80, "branch_0")]
        with ensembler.agent.override(model=FunctionModel(function=capturing_model)):
            ensembler.ensemble(_spec(), solutions, _strategy(), run_id="r5", round_index=0)
        assert "model_a.predict_proba" in captured["prompt"]
        assert "average the predicted probabilities" in captured["prompt"]
        assert "submission.csv" in captured["prompt"]
