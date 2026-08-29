"""End-to-end master pipeline integration test.

Verifies complete execution from a raw markdown task description through
the 5-stage ``MLEStarPipeline`` to production-ready ``./final/`` artifacts
on disk, and the CLI round-trip that copies them to ``--output``.
"""

import sys
from contextlib import ExitStack
from pathlib import Path

import pytest
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from problem_2_v2 import main
from problem_2_v2.config import MLEStarConfig
from problem_2_v2.execution.finalizer import FinalArtifact
from problem_2_v2.guardrails.leakage import DataLeakageCheckerAgent
from problem_2_v2.guardrails.usage import DataUsageCheckerAgent
from problem_2_v2.ingestion.extractor import TaskExtractor
from problem_2_v2.initialization.evaluator import CandidateEvaluatorAgent
from problem_2_v2.initialization.merger import ModelMergerAgent
from problem_2_v2.initialization.pipeline import InitializationPipeline
from problem_2_v2.orchestrator import MLEStarPipeline, MLEStarResult
from problem_2_v2.refinement.ablation import AblationAgent, AblationSummarizerAgent
from problem_2_v2.refinement.coder import CoderAgent
from problem_2_v2.refinement.extractor import CodeBlockExtractorAgent
from problem_2_v2.refinement.pipeline import RefinementPipeline
from problem_2_v2.refinement.planner import RefinementPlannerAgent
from problem_2_v2.runner.debugger import DebuggerAgent
from problem_2_v2.runner.sandbox import SubprocessRunner
from problem_2_v2.search.providers import MockSearchProvider, SearchResult
from problem_2_v2.search.retriever import RetrieverAgent

_MD = (
    "**Task Name:** Demo\n"
    "**Task Type:** TABULAR_CLASSIFICATION\n"
    "**Metric Name:** AUROC\n"
    "**Metric Direction:** MAXIMIZE\n"
    "**Target Variable:** label\n"
    "**Baseline Score:** 0.50\n"
    "**Dataset Files:** train.csv\n"
    "**Description:** classify.\n"
)

_CARD_ARGS = [
    {
        "model_name": "Model1",
        "rationale": "state of the art",
        "example_code": "import model1\nmodel = Model1()",
        "library_dependencies": ["model1"],
    }
]

FINAL_ENSEMBLE = "print('Final Validation Performance: 0.88')"

PRODUCTION_SCRIPT = (
    "from pathlib import Path\n"
    "import json\n"
    "final = Path('./final')\n"
    "final.mkdir(parents=True, exist_ok=True)\n"
    "(final / 'model.joblib').write_bytes(b'model')\n"
    "(final / 'submission.csv').write_text('id,pred\\n1,0.5\\n')\n"
    "(final / 'metrics.json').write_text(json.dumps({'auroc': 0.90}))\n"
    "print('Final Validation Performance: 0.90')\n"
)


def _report_args() -> dict[str, object]:
    return {
        "baseline_score": 0.50,
        "ablation_results": [
            {
                "variant_id": "model",
                "validation_score": 0.51,
                "delta_from_baseline": 0.01,
                "summary": "Model mattered most.",
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


def _planner_args() -> dict[str, object]:
    return {
        "method": "SIMPLE_AVERAGE",
        "natural_language_plan": "average the probabilities",
        "meta_learner_type": None,
    }


def _branch_factory(tmp_path: Path):
    """TestModel-configured branch factory producing 0.51/0.52 solutions."""

    def branch_builder(seed: int):
        score = 0.51 + seed * 0.01
        runner = SubprocessRunner(
            runs_dir=str(tmp_path / "runs"),
            timeout_seconds=5,
            python_executable=sys.executable,
        )
        debugger = DebuggerAgent(runner=runner, model="test", max_debug_rounds=1)
        branch_code = "print('Final Validation Performance: 0.50')\n"
        refined_block = f"model = 'seed{seed}'\nprint('Final Validation Performance: {score:.2f}')"
        target_block = "print('Final Validation Performance: 0.50')"

        def merger_model(messages, info):
            return ModelResponse(parts=[TextPart(content=branch_code)])

        init = InitializationPipeline(
            extractor=TaskExtractor(use_llm=False),
            retriever=RetrieverAgent(
                provider=MockSearchProvider(
                    results={
                        "classification": [
                            SearchResult(title="t", url="https://e.com", snippet="s")
                        ]
                    }
                ),
                model=TestModel(custom_output_args=_CARD_ARGS),
                num_candidates=1,
            ),
            evaluator=CandidateEvaluatorAgent(
                debugger=debugger, model=TestModel(custom_output_text=branch_code)
            ),
            merger=ModelMergerAgent(debugger=debugger, model=FunctionModel(function=merger_model)),
        )
        refine = RefinementPipeline(
            ablation=AblationAgent(model=TestModel(custom_output_text="print(1)")),
            summarizer=AblationSummarizerAgent(
                runner=runner, model=TestModel(custom_output_args=_report_args())
            ),
            extractor=CodeBlockExtractorAgent(
                model=TestModel(
                    custom_output_args=[
                        {
                            "code_block": target_block,
                            "plan": "use a seed",
                            "category": "MODEL_ARCHITECTURE",
                        }
                    ]
                )
            ),
            planner=RefinementPlannerAgent(model=TestModel(custom_output_text="plan")),
            coder=CoderAgent(
                model=TestModel(custom_output_text=f"```python\n{refined_block}\n```")
            ),
            leakage=DataLeakageCheckerAgent(model=TestModel(custom_output_args=_leak_clean_args())),
            usage=DataUsageCheckerAgent(
                model=TestModel(custom_output_text="All the provided information is used.")
            ),
            debugger=debugger,
            runner=runner,
            outer_loops=1,
            inner_loops=1,
        )
        return init, refine

    return branch_builder


def _pipeline(tmp_path: Path) -> MLEStarPipeline:
    return MLEStarPipeline(
        config=MLEStarConfig(
            runs_dir=str(tmp_path / "runs"),
            num_branches=2,
            ensemble_rounds=2,
            timeout_seconds=5,
        ),
        branch_builder=_branch_factory(tmp_path),
        search_provider=MockSearchProvider(),
    )


def _override_agents(pipeline: MLEStarPipeline) -> ExitStack:
    stack = ExitStack()
    stack.enter_context(
        pipeline.ensembler.agent.override(
            model=TestModel(custom_output_text=f"```python\n{FINAL_ENSEMBLE}\n```")
        )
    )
    stack.enter_context(
        pipeline.ensemble_pipeline.planner.agent.override(
            model=TestModel(custom_output_args=_planner_args())
        )
    )
    stack.enter_context(
        pipeline.execution.leakage.check_agent.override(
            model=TestModel(custom_output_args=_leak_clean_args())
        )
    )
    stack.enter_context(
        pipeline.execution.usage.agent.override(
            model=TestModel(custom_output_text="All the provided information is used.")
        )
    )
    stack.enter_context(
        pipeline.finalizer.agent.override(
            model=TestModel(custom_output_text=f"```python\n{PRODUCTION_SCRIPT}\n```")
        )
    )
    return stack


def _write_task(tmp_path: Path) -> tuple[Path, Path]:
    task_file = tmp_path / "problem.md"
    task_file.write_text(_MD, encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "train.csv").write_text("x,y\n1,0\n2,1\n3,0\n", encoding="utf-8")
    return task_file, data_dir


class TestEndToEndMaster:
    """Test the full markdown-to-artifact execution."""

    async def test_master_run_produces_final_artifacts(self, tmp_path: Path) -> None:
        task_file, data_dir = _write_task(tmp_path)
        pipeline = _pipeline(tmp_path)

        with _override_agents(pipeline):
            result = await pipeline.run_async(str(task_file), str(data_dir), run_id="e2e_master")

        assert isinstance(result, MLEStarResult)
        assert result.success is True
        assert len(result.branch_artifacts) == 2
        assert result.ensemble_result is not None
        assert result.final_artifact is not None
        assert result.final_score == pytest.approx(0.90)
        assert result.score_delta == pytest.approx(0.40)

        output = Path(result.final_artifact.output_dir)
        assert output.is_dir()
        assert (output / "submission.csv").exists()
        assert (output / "metrics.json").exists()
        assert (output / "model.joblib").exists()
        assert result.final_artifact.submission_path is not None
        assert result.final_artifact.metrics == {"auroc": 0.90}

    def test_cli_run_copies_artifacts_to_output(self, tmp_path: Path, capsys, monkeypatch) -> None:
        task_file, data_dir = _write_task(tmp_path)
        spec_dir = tmp_path / "prod"
        spec_dir.mkdir(exist_ok=True)
        (spec_dir / "submission.csv").write_text("id,pred\n1,0.5\n", encoding="utf-8")
        (spec_dir / "metrics.json").write_text('{"auroc": 0.9}', encoding="utf-8")
        (spec_dir / "model.joblib").write_bytes(b"model")

        from problem_2_v2.contracts.task import TaskSpecification
        from problem_2_v2.orchestrator import MLEStarPipeline as Orchestrator

        spec = TaskSpecification.from_markdown(_MD, dataset_dir=str(data_dir))

        def fake_run(self, task, data, run_id=None):
            return MLEStarResult(
                task_spec=spec,
                branch_artifacts=[],
                ensemble_result=None,
                final_artifact=FinalArtifact(
                    code="print(1)",
                    output_dir=str(spec_dir),
                    model_paths=[str(spec_dir / "model.joblib")],
                    metrics={"auroc": 0.9},
                    submission_path=str(spec_dir / "submission.csv"),
                    validation_score=0.9,
                    success=True,
                ),
                baseline_score=0.5,
                final_score=0.9,
                score_delta=0.4,
                duration_seconds=1.2,
                success=True,
            )

        monkeypatch.setattr(Orchestrator, "run", fake_run)
        out_dir = tmp_path / "final_out"
        code = main(
            [
                "run",
                "--task",
                str(task_file),
                "--data",
                str(data_dir),
                "--search-provider",
                "mock",
                "--output",
                str(out_dir),
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert (out_dir / "submission.csv").exists()
        assert (out_dir / "metrics.json").exists()
        assert (out_dir / "model.joblib").exists()
        assert "Artifacts written to" in out
