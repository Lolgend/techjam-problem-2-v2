"""Unit tests for the final artifact producer ($A_finalizer$).

Covers subsampling-stripping instructions, model serialization and
``metrics.json`` export discovery, ``./final/`` output structure, AST
validation, and the extended-timeout debugger fallback.
"""

import json
import sys
from pathlib import Path

import pytest
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.iteration import IterationLogEntry
from problem_2_v2.contracts.task import TaskSpecification
from problem_2_v2.execution.finalizer import FinalArtifact, FinalArtifactProducer
from problem_2_v2.execution.pipeline import ExecutionConfig
from problem_2_v2.runner.debugger import DebuggerAgent
from problem_2_v2.runner.sandbox import SubprocessRunner

WINNING_SOLUTION = (
    "import pandas as pd\n"
    "train = pd.read_csv('./input/train.csv')\n"
    "data = train.head(30000)\n"
    "model = LinearRegression()\n"
    "model.fit(data.drop(columns=['y']), data['y'])\n"
    "print('Final Validation Performance: 0.80')\n"
)

PRODUCTION_SCRIPT = (
    "from pathlib import Path\n"
    "import json\n"
    "final = Path('./final')\n"
    "final.mkdir(parents=True, exist_ok=True)\n"
    "(final / 'model.joblib').write_bytes(b'model')\n"
    "(final / 'submission.csv').write_text('id,pred\\n1,0.5\\n')\n"
    "(final / 'metrics.json').write_text(json.dumps({'auroc': 0.87, 'note': 'done'}))\n"
    "print('Final Validation Performance: 0.87')\n"
)

BROKEN_PRODUCTION = "def broken(:\n    pass\n"


def _spec() -> TaskSpecification:
    return TaskSpecification.from_markdown(
        "**Task Type:** TABULAR_CLASSIFICATION\n"
        "**Metric Name:** AUROC\n"
        "**Metric Direction:** MAXIMIZE\n"
        "**Target Variable:** y\n"
        "**Dataset Files:** train.csv\n",
        dataset_dir="/data",
    )


@pytest.fixture()
def finalizer(tmp_path: Path) -> FinalArtifactProducer:
    runner = SubprocessRunner(
        runs_dir=str(tmp_path / "runs"),
        timeout_seconds=5,
        python_executable=sys.executable,
    )
    debugger = DebuggerAgent(runner=runner, model="test", max_debug_rounds=1)
    return FinalArtifactProducer(
        debugger=debugger,
        model="test",
        config=ExecutionConfig(timeout_seconds=5),
    )


class TestFinalArtifactProducer:
    """Test full-data finalization to a production-ready artifact."""

    def test_produces_final_artifact(self, finalizer: FinalArtifactProducer) -> None:
        with finalizer.agent.override(
            model=TestModel(custom_output_text=f"```python\n{PRODUCTION_SCRIPT}\n```")
        ):
            artifact = finalizer.produce(WINNING_SOLUTION, _spec(), run_id="fin")
        assert isinstance(artifact, FinalArtifact)
        assert artifact.success is True
        assert artifact.validation_score == pytest.approx(0.87)
        assert artifact.metrics == {"auroc": 0.87}
        assert any("model.joblib" in p for p in artifact.model_paths)
        assert artifact.submission_path is not None
        assert Path(artifact.submission_path or "").exists()
        assert Path(artifact.output_dir).exists()
        assert "joblib" in artifact.code

    def test_prompt_removes_subsampling_and_serializes(
        self, finalizer: FinalArtifactProducer
    ) -> None:
        captured: dict[str, str] = {}

        def capturing_model(messages, info):
            captured["prompt"] = messages[-1].parts[0].content
            return ModelResponse(parts=[TextPart(content=f"```python\n{PRODUCTION_SCRIPT}\n```")])

        with finalizer.agent.override(model=FunctionModel(function=capturing_model)):
            finalizer.produce(WINNING_SOLUTION, _spec(), run_id="fin2")
        prompt = captured["prompt"].lower()
        assert "subsampl" in prompt
        assert "complete dataset" in prompt
        assert "model.joblib" in captured["prompt"] or "serialization" in prompt
        assert "metrics.json" in captured["prompt"]
        assert "submission.csv" in captured["prompt"]
        assert "head(30000)" in captured["prompt"]

    def test_debugger_recovers_broken_rewrite(self, finalizer: FinalArtifactProducer) -> None:
        with (
            finalizer.agent.override(
                model=TestModel(custom_output_text=f"```python\n{BROKEN_PRODUCTION}\n```")
            ),
            finalizer.debugger.agent.override(
                model=TestModel(custom_output_text=PRODUCTION_SCRIPT)
            ),
        ):
            artifact = finalizer.produce(WINNING_SOLUTION, _spec(), run_id="fin3")
        assert artifact.success is True
        assert artifact.validation_score == pytest.approx(0.87)

    def test_invalid_rewrite_produces_failure(self, finalizer: FinalArtifactProducer) -> None:
        with finalizer.agent.override(
            model=TestModel(custom_output_text="```python\nnot python (:\n```")
        ):
            artifact = finalizer.produce(WINNING_SOLUTION, _spec(), run_id="fin4")
        assert artifact.success is False
        assert artifact.validation_score is None

    def test_no_code_returns_failure_artifact(self, finalizer: FinalArtifactProducer) -> None:
        with finalizer.agent.override(model=TestModel(custom_output_text="I cannot help.")):
            artifact = finalizer.produce(WINNING_SOLUTION, _spec(), run_id="fin5")
        assert artifact.success is False
        assert artifact.validation_score is None
        assert artifact.model_paths == []

    def test_config_drives_production_timeout(self) -> None:
        producer = FinalArtifactProducer(
            model="test",
            config=ExecutionConfig(production_timeout_seconds=7200),
        )
        assert producer.debugger.runner.timeout_seconds == 7200

    def test_metrics_parsing_skips_non_numeric(self, tmp_path: Path) -> None:
        producer = FinalArtifactProducer(model="test", config=ExecutionConfig(timeout_seconds=5))
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps({"auroc": 0.87, "note": "done"}), encoding="utf-8")
        parsed = producer._load_metrics(metrics_file)
        assert parsed == {"auroc": 0.87}


class TestFinalizerIterationLogging:
    """Test Stage 4 finalizer records in the unified iteration log."""

    def test_produce_logs_finalization_entry(self, finalizer: FinalArtifactProducer) -> None:
        with finalizer.agent.override(
            model=TestModel(custom_output_text=f"```python\n{PRODUCTION_SCRIPT}\n```")
        ):
            artifact = finalizer.produce(WINNING_SOLUTION, _spec(), run_id="finlog")
        assert artifact.success is True
        log_path = Path(finalizer.debugger.runner.runs_dir) / "finlog" / "iteration_logs.jsonl"
        assert log_path.is_file()
        entries = [
            IterationLogEntry.model_validate_json(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
        ]
        assert len(entries) == 1
        entry = entries[0]
        assert entry.iteration_id == "final_prod"
        assert entry.stage == "FINALIZATION"
        assert entry.success is True
        assert entry.validation_score == pytest.approx(0.87)
        assert entry.metrics == {"auroc": 0.87}
        assert entry.code_diff != ""
        assert entry.target_component == "FINAL_PRODUCTION"
        assert entry.delta_from_baseline is None
        assert entry.branch_index is None


class TestFinalizerSubmissionSchema:
    """Test that the finalizer enforces the submit.py 4-column submission schema."""

    def test_instructions_mandate_submission_schema(self) -> None:
        from problem_2_v2.execution.finalizer import _FINALIZER_INSTRUCTIONS

        assert "row_id,user_id,video_id,score" in _FINALIZER_INSTRUCTIONS
        assert "submit.py" in _FINALIZER_INSTRUCTIONS
        assert "data.load" in _FINALIZER_INSTRUCTIONS

    def test_build_prompt_mandates_submission_schema(self, finalizer) -> None:
        prompt = finalizer.build_prompt(WINNING_SOLUTION, _spec())
        assert "row_id,user_id,video_id,score" in prompt
        assert "submit.py" in prompt
        assert "deterministic" in prompt
        assert "data.load()" in prompt

    def test_sent_prompt_includes_submission_schema(self, finalizer) -> None:
        captured: dict[str, str] = {}

        def capturing_model(messages, info):
            captured["prompt"] = messages[-1].parts[0].content
            return ModelResponse(parts=[TextPart(content=f"```python\n{PRODUCTION_SCRIPT}\n```")])

        with finalizer.agent.override(model=FunctionModel(function=capturing_model)):
            finalizer.produce(WINNING_SOLUTION, _spec(), run_id="fin_schema")
        prompt = captured["prompt"]
        assert "row_id,user_id,video_id,score" in prompt
        assert "submit.py" in prompt
        assert "deterministic" in prompt

    def test_instructions_mandate_evaluate_harness(self) -> None:
        from problem_2_v2.execution.finalizer import _FINALIZER_INSTRUCTIONS

        assert "from evaluate import evaluate" in _FINALIZER_INSTRUCTIONS

    def test_build_prompt_mandates_evaluate_harness(self, finalizer) -> None:
        prompt = finalizer.build_prompt(WINNING_SOLUTION, _spec())
        assert "from evaluate import evaluate" in prompt
