"""Unit tests for the master orchestrator and configuration.

Covers ``MLEStarConfig`` defaults and validation, ``MLEStarResult``
fields, the 5-stage coordination of ``MLEStarPipeline``, baseline delta
calculation, and dry-run path validation.
"""

import sys
from contextlib import ExitStack
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from problem_2_v2.config import MLEStarConfig
from problem_2_v2.contracts.task import TaskSpecification
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
    """Build a TestModel-configured branch factory for orchestrator tests."""

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


def _write_task(tmp_path: Path) -> tuple[Path, Path]:
    task_file = tmp_path / "problem.md"
    task_file.write_text(_MD, encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "train.csv").write_text("x,y\n1,0\n2,1\n3,0\n", encoding="utf-8")
    return task_file, data_dir


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


class TestMLEStarConfig:
    """Test the master configuration hyperparameters."""

    def test_defaults(self) -> None:
        config = MLEStarConfig()
        assert config.model == "openai:gpt-4o"
        assert config.search_provider == "websearch"
        assert config.num_candidates == 4
        assert config.num_branches == 2
        assert config.outer_loops == 3
        assert config.inner_loops == 3
        assert config.ensemble_rounds == 3
        assert config.seeds is None
        assert config.subsample_size == 30000
        assert config.timeout_seconds == 600
        assert config.production_timeout_seconds == 3600
        assert config.max_debug_rounds == 3
        assert config.runs_dir == "runs"
        assert config.final_output_dir == "final"

    def test_overrides(self) -> None:
        config = MLEStarConfig(
            model="google:gemini-2.0-flash",
            search_provider="mock",
            num_candidates=8,
            num_branches=4,
            seeds=[7, 8, 9],
            timeout_seconds=120,
        )
        assert config.model == "google:gemini-2.0-flash"
        assert config.search_provider == "mock"
        assert config.num_candidates == 8
        assert config.num_branches == 4
        assert config.seeds == [7, 8, 9]
        assert config.timeout_seconds == 120

    def test_invalid_search_provider_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MLEStarConfig(search_provider="bogus")

    def test_non_positive_counts_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MLEStarConfig(num_branches=0)

    def test_zero_ensemble_rounds_accepted(self) -> None:
        config = MLEStarConfig(ensemble_rounds=0)
        assert config.ensemble_rounds == 0


class TestMLEStarPipeline:
    """Test the 5-stage master coordination."""

    def test_build_provider_websearch_returns_none(self, tmp_path: Path) -> None:
        pipeline = MLEStarPipeline(
            config=MLEStarConfig(runs_dir=str(tmp_path / "runs"), search_provider="websearch")
        )
        assert pipeline._provider is None

    def test_build_provider_builtin_returns_none(self, tmp_path: Path) -> None:
        pipeline = MLEStarPipeline(
            config=MLEStarConfig(runs_dir=str(tmp_path / "runs"), search_provider="builtin")
        )
        assert pipeline._provider is None

    def test_build_provider_mock(self, tmp_path: Path) -> None:
        pipeline = MLEStarPipeline(
            config=MLEStarConfig(runs_dir=str(tmp_path / "runs"), search_provider="mock")
        )
        assert pipeline._provider is not None
        assert pipeline._provider.provider_name == "mock"

    def test_build_provider_tavily_requires_key(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        with pytest.raises(ValueError):
            MLEStarPipeline(
                config=MLEStarConfig(runs_dir=str(tmp_path / "runs"), search_provider="tavily")
            )

    def test_build_provider_google_requires_key(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)
        with pytest.raises(ValueError):
            MLEStarPipeline(
                config=MLEStarConfig(runs_dir=str(tmp_path / "runs"), search_provider="google")
            )

    def test_build_provider_duckduckgo(self, tmp_path: Path) -> None:
        pipeline = MLEStarPipeline(
            config=MLEStarConfig(runs_dir=str(tmp_path / "runs"), search_provider="duckduckgo")
        )
        assert pipeline._provider is not None
        assert pipeline._provider.provider_name == "duckduckgo"

    def test_default_branch_builder(self, tmp_path: Path) -> None:
        pipeline = MLEStarPipeline(
            config=MLEStarConfig(runs_dir=str(tmp_path / "runs"), timeout_seconds=5),
        )
        init, refine = pipeline._build_branch(0)
        assert isinstance(init, InitializationPipeline)
        assert isinstance(refine, RefinementPipeline)
        assert init.retriever.num_candidates == 4
        assert init.retriever.provider is None
        assert any(isinstance(c, WebSearch) for c in init.retriever.capabilities)

    def test_dry_run_validation(self, tmp_path: Path) -> None:
        task_file, data_dir = _write_task(tmp_path)
        pipeline = _pipeline(tmp_path)

        spec = pipeline.validate(str(task_file), str(data_dir))
        assert isinstance(spec, TaskSpecification)
        assert spec.task_name == "Demo"
        assert spec.baseline_score == pytest.approx(0.50)

    def test_dry_run_rejects_missing_paths(self, tmp_path: Path) -> None:
        pipeline = _pipeline(tmp_path)
        with pytest.raises(FileNotFoundError):
            pipeline.validate(str(tmp_path / "missing.md"), str(tmp_path))
        task_file = tmp_path / "problem.md"
        task_file.write_text(_MD, encoding="utf-8")
        with pytest.raises(NotADirectoryError):
            pipeline.validate(str(task_file), str(tmp_path / "no_data"))

    async def test_five_stage_coordination(self, tmp_path: Path) -> None:
        task_file, data_dir = _write_task(tmp_path)
        pipeline = _pipeline(tmp_path)

        with _override_agents(pipeline):
            result = await pipeline.run_async(str(task_file), str(data_dir), run_id="master1")

        assert isinstance(result, MLEStarResult)
        assert result.success is True
        assert result.task_spec.baseline_score == pytest.approx(0.50)
        assert len(result.branch_artifacts) == 2
        assert result.ensemble_result is not None
        assert result.ensemble_result.best_score == pytest.approx(0.88)
        assert result.final_artifact is not None
        assert result.final_artifact.validation_score == pytest.approx(0.90)
        assert result.final_score == pytest.approx(0.90)
        assert result.score_delta == pytest.approx(0.40)
        assert result.duration_seconds >= 0

    def test_run_sync_entrypoint(self, tmp_path: Path) -> None:
        task_file, data_dir = _write_task(tmp_path)
        pipeline = _pipeline(tmp_path)

        with _override_agents(pipeline):
            result = pipeline.run(str(task_file), str(data_dir), run_id="master2")
        assert result.success is True
        assert result.score_delta == pytest.approx(0.40)

    async def test_failed_branches_produce_failure_result(self, tmp_path: Path) -> None:
        task_file, data_dir = _write_task(tmp_path)

        def exploding_model(messages, info):
            raise RuntimeError("evaluator backend down")

        def failing_branch_builder(seed: int):
            runner = SubprocessRunner(
                runs_dir=str(tmp_path / "runs"),
                timeout_seconds=5,
                python_executable=sys.executable,
            )
            debugger = DebuggerAgent(runner=runner, model="test", max_debug_rounds=1)
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
                    debugger=debugger, model=FunctionModel(function=exploding_model)
                ),
                merger=ModelMergerAgent(debugger=debugger, model="test"),
            )
            refine = RefinementPipeline(
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
                inner_loops=1,
            )
            return init, refine

        pipeline = MLEStarPipeline(
            config=MLEStarConfig(
                runs_dir=str(tmp_path / "runs"), num_branches=2, timeout_seconds=5
            ),
            branch_builder=failing_branch_builder,
            search_provider=MockSearchProvider(),
        )
        result = await pipeline.run_async(str(task_file), str(data_dir), run_id="master3")
        assert result.success is False
        assert result.branch_artifacts == []
        assert result.ensemble_result is None
        assert result.final_artifact is None
        assert result.score_delta is None

    async def test_single_branch_skips_ensembling(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        task_file, data_dir = _write_task(tmp_path)
        pipeline = MLEStarPipeline(
            config=MLEStarConfig(
                runs_dir=str(tmp_path / "runs"),
                num_branches=1,
                ensemble_rounds=2,
                timeout_seconds=5,
            ),
            branch_builder=_branch_factory(tmp_path),
            search_provider=MockSearchProvider(),
        )
        captured: dict[str, str] = {}
        original = pipeline.finalizer.produce

        def spy_produce(
            code: str, spec: TaskSpecification, run_id: str = "finalize"
        ) -> FinalArtifact:
            captured["code"] = code
            captured["run_id"] = run_id
            return original(code, spec, run_id)

        monkeypatch.setattr(pipeline.finalizer, "produce", spy_produce)

        with _override_agents(pipeline):
            result = await pipeline.run_async(str(task_file), str(data_dir), run_id="single1")

        out = capsys.readouterr().out
        assert result.success is True
        assert result.ensemble_result is None
        assert result.final_artifact is not None
        assert result.final_score == pytest.approx(0.90)
        assert "Adaptive Ensembling skipped (single candidate" in out
        assert "seed0" in captured["code"]
        assert captured["run_id"] == "single1/final"

    async def test_zero_ensemble_rounds_skips_ensembling(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        task_file, data_dir = _write_task(tmp_path)
        pipeline = MLEStarPipeline(
            config=MLEStarConfig(
                runs_dir=str(tmp_path / "runs"),
                num_branches=2,
                ensemble_rounds=0,
                timeout_seconds=5,
            ),
            branch_builder=_branch_factory(tmp_path),
            search_provider=MockSearchProvider(),
        )
        captured: dict[str, str] = {}
        original = pipeline.finalizer.produce

        def spy_produce(
            code: str, spec: TaskSpecification, run_id: str = "finalize"
        ) -> FinalArtifact:
            captured["code"] = code
            return original(code, spec, run_id)

        monkeypatch.setattr(pipeline.finalizer, "produce", spy_produce)

        with _override_agents(pipeline):
            result = await pipeline.run_async(str(task_file), str(data_dir), run_id="zero_rounds")

        out = capsys.readouterr().out
        assert result.success is True
        assert result.ensemble_result is None
        assert result.final_artifact is not None
        assert "Adaptive Ensembling skipped (ensemble_rounds=0" in out
        assert "seed1" in captured["code"]
