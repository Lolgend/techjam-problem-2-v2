"""Unit tests for the parallel solution generator."""

import sys
from pathlib import Path

import pytest
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.task import PipelineArtifact
from problem_2_v2.ensembling.parallel import ParallelSolutionGenerator
from problem_2_v2.guardrails.leakage import DataLeakageCheckerAgent
from problem_2_v2.guardrails.usage import DataUsageCheckerAgent
from problem_2_v2.ingestion.extractor import TaskExtractor
from problem_2_v2.initialization.evaluator import CandidateEvaluatorAgent
from problem_2_v2.initialization.merger import ModelMergerAgent
from problem_2_v2.initialization.pipeline import InitializationPipeline
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


def build_branch_factory(tmp_path: Path, seed_scores: dict[int, float]):
    """Return a branch builder producing score-per-seed pipelines.

    Agents are constructed with TestModel/FunctionModel *instances* so the
    mocked behavior persists beyond any context-manager scope (branches run
    in worker threads via ``asyncio.to_thread``).
    """

    def branch_builder(seed: int) -> tuple[InitializationPipeline, RefinementPipeline]:
        score = seed_scores.get(seed, 0.50)
        runs_dir = str(tmp_path / "runs")
        runner = SubprocessRunner(
            runs_dir=runs_dir,
            timeout_seconds=5,
            python_executable=sys.executable,
        )
        debugger = DebuggerAgent(runner=runner, model="test", max_debug_rounds=1)
        branch_code = "model = 'base'\nprint('Final Validation Performance: 0.50')\n"
        refined_block = f"model = 'seed{seed}'\nprint('Final Validation Performance: {score:.2f}')"
        target_block = "model = 'base'\nprint('Final Validation Performance: 0.50')"

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


class TestParallelSolutionGenerator:
    """Test concurrent multi-seed branch execution."""

    def test_generates_l_artifacts_concurrently(self, tmp_path: Path) -> None:
        factory = build_branch_factory(tmp_path, {0: 0.51, 1: 0.52})
        generator = ParallelSolutionGenerator(branch_builder=factory, num_branches=2)

        async def run() -> list[PipelineArtifact]:
            return await generator.generate(_MD, dataset_dir="/data", run_id="par", seeds=[0, 1])

        artifacts = asyncio_run(run())
        assert len(artifacts) == 2
        assert artifacts[0].validation_score == pytest.approx(0.51)
        assert artifacts[1].validation_score == pytest.approx(0.52)
        assert artifacts[0].full_code != artifacts[1].full_code
        assert "seed0" in artifacts[0].full_code
        assert "seed1" in artifacts[1].full_code

    def test_branches_use_isolated_run_ids(self, tmp_path: Path) -> None:
        factory = build_branch_factory(tmp_path, {0: 0.51, 1: 0.52})
        generator = ParallelSolutionGenerator(branch_builder=factory, num_branches=2)

        async def run() -> list[PipelineArtifact]:
            return await generator.generate(_MD, dataset_dir="/data", run_id="par2", seeds=[0, 1])

        artifacts = asyncio_run(run())
        assert "branch_0" in artifacts[0].iteration_stage
        assert "branch_1" in artifacts[1].iteration_stage
        assert (tmp_path / "runs" / "par2" / "branch_0").exists()
        assert (tmp_path / "runs" / "par2" / "branch_1").exists()

    def test_branch_failure_is_isolated(self, tmp_path: Path) -> None:
        factory = build_branch_factory(tmp_path, {0: 0.51, 1: 0.52})

        def failing_builder(seed: int) -> tuple[InitializationPipeline, RefinementPipeline]:
            if seed == 1:
                raise RuntimeError("branch setup failed")
            return factory(seed)

        generator = ParallelSolutionGenerator(branch_builder=failing_builder, num_branches=2)

        async def run() -> list[PipelineArtifact]:
            return await generator.generate(_MD, dataset_dir="/data", run_id="par3", seeds=[0, 1])

        artifacts = asyncio_run(run())
        assert len(artifacts) == 1
        assert artifacts[0].validation_score == pytest.approx(0.51)

    def test_default_seeds_are_distinct(self, tmp_path: Path) -> None:
        factory = build_branch_factory(tmp_path, {0: 0.51, 1: 0.52})
        generator = ParallelSolutionGenerator(branch_builder=factory, num_branches=2)

        async def run() -> list[PipelineArtifact]:
            return await generator.generate(_MD, dataset_dir="/data", run_id="par4")

        artifacts = asyncio_run(run())
        assert len(artifacts) == 2

    def test_duplicate_seeds_are_rejected(self, tmp_path: Path) -> None:
        factory = build_branch_factory(tmp_path, {0: 0.51})
        generator = ParallelSolutionGenerator(branch_builder=factory, num_branches=2)

        async def run() -> list[PipelineArtifact]:
            return await generator.generate(_MD, dataset_dir="/data", run_id="par5", seeds=[0, 0])

        with pytest.raises(ValueError, match="distinct"):
            asyncio_run(run())

    def test_branch_artifacts_keep_lineage_diff(self, tmp_path: Path) -> None:
        factory = build_branch_factory(tmp_path, {0: 0.51, 1: 0.52})
        generator = ParallelSolutionGenerator(branch_builder=factory, num_branches=2)

        async def run() -> list[PipelineArtifact]:
            return await generator.generate(_MD, dataset_dir="/data", run_id="par6", seeds=[0, 1])

        artifacts = asyncio_run(run())
        assert all(artifact.applied_diff is not None for artifact in artifacts)


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)
