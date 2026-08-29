"""End-to-end integration test: guardrail -> execution -> finalization chain.

Runs a real subprocess chain: a broken candidate script is repaired by the
debugger inside ``ExecutionGuardrailPipeline``, then the executed winning
solution is finalized by ``FinalArtifactProducer`` into a ``./final/``
production artifact (model, ``metrics.json``, ``submission.csv``).
"""

import sys
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.task import TaskSpecification
from problem_2_v2.execution.finalizer import FinalArtifactProducer
from problem_2_v2.execution.pipeline import ExecutionConfig, ExecutionGuardrailPipeline
from problem_2_v2.guardrails.leakage import DataLeakageCheckerAgent
from problem_2_v2.guardrails.usage import DataUsageCheckerAgent
from problem_2_v2.runner.debugger import DebuggerAgent
from problem_2_v2.runner.sandbox import SubprocessRunner

BROKEN_CANDIDATE = "def broken(:\n    pass\n"

REPAIRED_CANDIDATE = (
    "from pathlib import Path\n"
    "rows = Path('./input/train.csv').read_text().splitlines()\n"
    "print('Final Validation Performance: 0.80')\n"
)

PRODUCTION_SCRIPT = (
    "from pathlib import Path\n"
    "import json\n"
    "final = Path('./final')\n"
    "final.mkdir(parents=True, exist_ok=True)\n"
    "(final / 'model.joblib').write_bytes(b'model')\n"
    "(final / 'submission.csv').write_text('id,pred\\n1,0.5\\n')\n"
    "(final / 'metrics.json').write_text(json.dumps({'auroc': 0.87}))\n"
    "print('Final Validation Performance: 0.87')\n"
)


def _leak_clean_args() -> dict[str, object]:
    return {
        "leakage_status": "No Data Leakage",
        "is_leaking": False,
        "suspicious_code_block": None,
        "corrected_code_block": None,
        "explanation": "clean",
    }


class TestEndToEndExecution:
    """Test the complete guardrail -> execution -> finalization chain."""

    def test_candidate_to_final_artifact_chain(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train.csv").write_text("x,y\n1,0\n2,1\n3,0\n", encoding="utf-8")

        runner = SubprocessRunner(
            runs_dir=str(tmp_path / "runs"),
            timeout_seconds=10,
            python_executable=sys.executable,
        )
        leakage = DataLeakageCheckerAgent(model="test")
        usage = DataUsageCheckerAgent(model="test")
        debugger = DebuggerAgent(runner=runner, model="test", max_debug_rounds=2)
        pipeline = ExecutionGuardrailPipeline(
            leakage=leakage,
            usage=usage,
            runner=runner,
            debugger=debugger,
        )
        finalizer = FinalArtifactProducer(
            debugger=debugger,
            model="test",
            config=ExecutionConfig(timeout_seconds=10),
        )
        spec = TaskSpecification.from_markdown(
            "**Task Type:** TABULAR_CLASSIFICATION\n"
            "**Metric Name:** AUROC\n"
            "**Metric Direction:** MAXIMIZE\n"
            "**Target Variable:** y\n"
            "**Dataset Files:** train.csv\n",
            dataset_dir=str(data_dir),
        )

        with (
            leakage.check_agent.override(model=TestModel(custom_output_args=_leak_clean_args())),
            usage.agent.override(
                model=TestModel(custom_output_text="All the provided information is used.")
            ),
            debugger.agent.override(model=TestModel(custom_output_text=REPAIRED_CANDIDATE)),
            finalizer.agent.override(
                model=TestModel(custom_output_text=f"```python\n{PRODUCTION_SCRIPT}\n```")
            ),
        ):
            result = pipeline.run(BROKEN_CANDIDATE, spec, run_id="e2e", candidate_id="cand")
            assert result.success is True
            assert result.validation_score == pytest.approx(0.80)
            assert pipeline.last_debug_rounds == 1
            assert pipeline.last_executed_code == REPAIRED_CANDIDATE

            artifact = finalizer.produce(pipeline.last_executed_code or "", spec, run_id="e2e")

        assert artifact.success is True
        assert artifact.validation_score == pytest.approx(0.87)
        assert artifact.metrics == {"auroc": 0.87}
        assert len(artifact.model_paths) == 1
        assert artifact.submission_path is not None
        output = Path(artifact.output_dir)
        assert (output / "submission.csv").exists()
        assert (output / "metrics.json").exists()
        assert "sandbox_final" in str(output)
