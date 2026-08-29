"""Integration tests for the end-to-end initialization pipeline."""

import sys
from pathlib import Path

import pytest
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from problem_2_v2.ingestion.extractor import TaskExtractor
from problem_2_v2.initialization.evaluator import CandidateEvaluatorAgent
from problem_2_v2.initialization.merger import ModelMergerAgent
from problem_2_v2.initialization.pipeline import InitializationPipeline, InitializationResult
from problem_2_v2.runner.debugger import DebuggerAgent
from problem_2_v2.runner.sandbox import SubprocessRunner
from problem_2_v2.search.providers import MockSearchProvider, SearchResult
from problem_2_v2.search.retriever import RetrieverAgent

_MD = (
    "**Task Name:** Demo CTR\n"
    "**Task Type:** RECOMMENDER_RANKING\n"
    "**Metric Name:** NDCG@10\n"
    "**Metric Direction:** MAXIMIZE\n"
    "**Target Variable:** is_click\n"
    "**Baseline Score:** 0.50\n"
    "**Description:** Predict clicks on short videos.\n"
)

_CARD_ARGS = [
    {
        "model_name": f"Model{i}",
        "rationale": "state of the art",
        "example_code": f"import model{i}\nmodel = Model{i}()",
        "library_dependencies": [f"model{i}"],
    }
    for i in range(1, 3)
]

_FINAL_SCORE = "print('Final Validation Performance: 0.55')"


def _merger_model(messages, info):
    return ModelResponse(parts=[TextPart(content=_FINAL_SCORE)])


@pytest.fixture()
def pipeline(tmp_path: Path) -> InitializationPipeline:
    runner = SubprocessRunner(
        runs_dir=str(tmp_path / "runs"),
        timeout_seconds=5,
        python_executable=sys.executable,
    )
    debugger = DebuggerAgent(runner=runner, model="test", max_debug_rounds=1)
    extractor = TaskExtractor(use_llm=False)
    retriever = RetrieverAgent(
        provider=MockSearchProvider(
            results={
                "ranking": [
                    SearchResult(title="t", url="https://e.com", snippet="s"),
                ]
            }
        ),
        model="test",
        num_candidates=2,
    )
    evaluator = CandidateEvaluatorAgent(debugger=debugger, model="test")
    merger = ModelMergerAgent(debugger=debugger, model="test")
    return InitializationPipeline(
        extractor=extractor,
        retriever=retriever,
        evaluator=evaluator,
        merger=merger,
    )


class TestInitializationPipeline:
    """Test the wired extractor -> retriever -> evaluator -> merger flow."""

    def test_run_completes_end_to_end(self, pipeline: InitializationPipeline) -> None:
        with (
            pipeline.retriever.agent.override(model=TestModel(custom_output_args=_CARD_ARGS)),
            pipeline.evaluator.agent.override(model=TestModel(custom_output_text=_FINAL_SCORE)),
            pipeline.merger.agent.override(model=FunctionModel(function=_merger_model)),
        ):
            result = pipeline.run(_MD, dataset_dir="/data", run_id="e2e")

        assert isinstance(result, InitializationResult)
        assert result.task.task_type.value == "RECOMMENDER_RANKING"
        assert len(result.candidates.candidates) == 2
        assert len(result.evaluations) == 2
        assert result.outcome.final_score is not None
        assert result.outcome.final_code != ""
        assert result.best_score == result.outcome.final_score
        assert result.best_code == result.outcome.final_code

    def test_run_uses_mock_provider_queries(self, pipeline: InitializationPipeline) -> None:
        with (
            pipeline.retriever.agent.override(model=TestModel(custom_output_args=_CARD_ARGS)),
            pipeline.evaluator.agent.override(model=TestModel(custom_output_text=_FINAL_SCORE)),
            pipeline.merger.agent.override(model=FunctionModel(function=_merger_model)),
        ):
            result = pipeline.run(_MD, dataset_dir="/data", run_id="e2e2")
        assert "NDCG@10" in result.candidates.query_used
