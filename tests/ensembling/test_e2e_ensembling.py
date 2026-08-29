"""End-to-end integration test: parallel generation -> adaptive ensembling."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.task import PipelineArtifact
from problem_2_v2.ensembling.ensembler import EnsemblerAgent
from problem_2_v2.ensembling.parallel import ParallelSolutionGenerator
from problem_2_v2.ensembling.pipeline import EnsemblePipeline
from problem_2_v2.ensembling.planner import EnsemblePlannerAgent
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

FINAL_ENSEMBLE = "print('Final Validation Performance: 0.90')"
BEST_INDIVIDUAL = "model = 'seed1'\nprint('Final Validation Performance: 0.52')"


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


def _planner_args(plan: str) -> dict[str, object]:
    return {
        "method": "SIMPLE_AVERAGE",
        "natural_language_plan": plan,
        "meta_learner_type": None,
    }


def build_branch_factory(tmp_path: Path):
    """Seed-aware branch builder producing 0.51/0.52 solutions."""

    def branch_builder(seed: int) -> tuple[InitializationPipeline, RefinementPipeline]:
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


class TestEndToEndEnsembling:
    """Test L candidates through R ensembling rounds to the final solution."""

    def test_parallel_generation_feeds_ensembling(self, tmp_path: Path) -> None:
        generator = ParallelSolutionGenerator(
            branch_builder=build_branch_factory(tmp_path), num_branches=2
        )
        runner = SubprocessRunner(
            runs_dir=str(tmp_path / "runs"),
            timeout_seconds=5,
            python_executable=sys.executable,
        )
        debugger = DebuggerAgent(runner=runner, model="test", max_debug_rounds=1)
        pipeline = EnsemblePipeline(
            planner=EnsemblePlannerAgent(model="test"),
            ensembler=EnsemblerAgent(debugger=debugger, model="test"),
            runner=runner,
            rounds=2,
        )

        async def workflow() -> tuple[list[PipelineArtifact], object]:
            artifacts = await generator.generate(
                _MD, dataset_dir="/data", run_id="e2e", seeds=[0, 1]
            )
            with (
                pipeline.planner.agent.override(
                    model=TestModel(custom_output_args=_planner_args("average the probabilities"))
                ),
                pipeline.ensembler.agent.override(
                    model=TestModel(custom_output_text=f"```python\n{FINAL_ENSEMBLE}\n```")
                ),
            ):
                # run() is a sync pipeline: offload to a worker thread so its
                # agent.run_sync calls work outside the running event loop.
                result = await asyncio.to_thread(pipeline.run, _spec(), artifacts, "e2e")
            return artifacts, result

        artifacts, result = asyncio.run(workflow())
        assert len(artifacts) == 2
        assert result.best_score == pytest.approx(0.90)
        assert "0.90" in result.best_code
        logs_path = Path(result.logs_path or "")
        records = [json.loads(line) for line in logs_path.read_text(encoding="utf-8").splitlines()]
        assert len(records) == 2


def _spec():
    """Build a task spec; the pipeline only needs the metric direction."""
    from problem_2_v2.contracts.task import TaskSpecification

    return TaskSpecification.from_markdown(
        "**Task Type:** TABULAR_CLASSIFICATION\n"
        "**Metric Name:** AUROC\n"
        "**Metric Direction:** MAXIMIZE\n"
        "**Target Variable:** label\n"
        "**Dataset Files:** train.csv\n",
        dataset_dir="/data",
    )
