"""End-to-end integration test: initialization output feeds refinement."""

import sys
from pathlib import Path

import pytest
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from problem_2_v2.guardrails.leakage import DataLeakageCheckerAgent
from problem_2_v2.guardrails.usage import DataUsageCheckerAgent
from problem_2_v2.ingestion.extractor import TaskExtractor
from problem_2_v2.initialization.evaluator import CandidateEvaluatorAgent
from problem_2_v2.initialization.merger import ModelMergerAgent
from problem_2_v2.initialization.pipeline import InitializationPipeline, InitializationResult
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

INIT_CODE = "x = 1\nmodel = 'a'\nprint('Final Validation Performance: 0.55')\n"
TARGET_BLOCK = "model = 'a'\nprint('Final Validation Performance: 0.55')"
REFINED_BLOCK = "model = 'b'\nprint('Final Validation Performance: 0.60')"
FINAL_CODE = "x = 1\nmodel = 'b'\nprint('Final Validation Performance: 0.60')"

_CARD_ARGS = [
    {
        "model_name": f"Model{i}",
        "rationale": "state of the art",
        "example_code": f"import model{i}\nmodel = Model{i}()",
        "library_dependencies": [f"model{i}"],
    }
    for i in range(1, 3)
]


def _merger_model(messages, info):
    return ModelResponse(parts=[TextPart(content=INIT_CODE)])


@pytest.fixture()
def runner(tmp_path: Path) -> SubprocessRunner:
    return SubprocessRunner(
        runs_dir=str(tmp_path / "runs"),
        timeout_seconds=5,
        python_executable=sys.executable,
    )


@pytest.fixture()
def init_pipeline(runner: SubprocessRunner) -> InitializationPipeline:
    debugger = DebuggerAgent(runner=runner, model="test", max_debug_rounds=1)
    return InitializationPipeline(
        extractor=TaskExtractor(use_llm=False),
        retriever=RetrieverAgent(
            provider=MockSearchProvider(
                results={
                    "classification": [SearchResult(title="t", url="https://e.com", snippet="s")]
                }
            ),
            model="test",
            num_candidates=2,
        ),
        evaluator=CandidateEvaluatorAgent(debugger=debugger, model="test"),
        merger=ModelMergerAgent(debugger=debugger, model="test"),
    )


@pytest.fixture()
def refine_pipeline(runner: SubprocessRunner) -> RefinementPipeline:
    debugger = DebuggerAgent(runner=runner, model="test", max_debug_rounds=1)
    return RefinementPipeline(
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


def _report_args() -> dict[str, object]:
    return {
        "baseline_score": 0.55,
        "ablation_results": [
            {
                "variant_id": "model",
                "validation_score": 0.57,
                "delta_from_baseline": 0.02,
                "summary": "Model mattered most.",
            }
        ],
        "highest_impact_component": "model",
        "raw_log_summary": "...",
    }


class TestEndToEndRefinement:
    """Test initialization result flowing into the refinement pipeline."""

    def test_initialization_feeds_refinement(self, init_pipeline, refine_pipeline) -> None:
        with (
            init_pipeline.retriever.agent.override(model=TestModel(custom_output_args=_CARD_ARGS)),
            init_pipeline.evaluator.agent.override(model=TestModel(custom_output_text=INIT_CODE)),
            init_pipeline.merger.agent.override(model=FunctionModel(function=_merger_model)),
        ):
            init_result = init_pipeline.run(_MD, dataset_dir="/data", run_id="e2e_init")

        assert isinstance(init_result, InitializationResult)
        assert init_result.best_score == pytest.approx(0.55)

        with (
            refine_pipeline.ablation.agent.override(
                model=TestModel(custom_output_text="print('ablation done')")
            ),
            refine_pipeline.summarizer.agent.override(
                model=TestModel(custom_output_args=_report_args())
            ),
            refine_pipeline.extractor.agent.override(
                model=TestModel(
                    custom_output_args=[
                        {
                            "code_block": TARGET_BLOCK,
                            "plan": "try a different model",
                            "category": "MODEL_ARCHITECTURE",
                        }
                    ]
                )
            ),
            refine_pipeline.coder.agent.override(
                model=TestModel(custom_output_text=f"```python\n{REFINED_BLOCK}\n```")
            ),
            refine_pipeline.leakage.check_agent.override(
                model=TestModel(
                    custom_output_args={
                        "leakage_status": "No Data Leakage",
                        "is_leaking": False,
                        "suspicious_code_block": None,
                        "corrected_code_block": None,
                        "explanation": "clean",
                    }
                )
            ),
            refine_pipeline.usage.agent.override(
                model=TestModel(custom_output_text="All the provided information is used.")
            ),
        ):
            refine_result = refine_pipeline.refine(
                init_result.task,
                initial_code=init_result.best_code,
                initial_score=init_result.best_score,
                run_id="e2e_refine",
            )

        assert refine_result.final_score == pytest.approx(0.60)
        assert refine_result.final_code == FINAL_CODE
        assert refine_result.logs_path is not None
        assert Path(refine_result.logs_path).exists()
