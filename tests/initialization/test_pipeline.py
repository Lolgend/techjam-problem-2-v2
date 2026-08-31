"""Integration tests for the end-to-end initialization pipeline."""

import json
import sys
from pathlib import Path

import pytest
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.task import TaskSpecification
from problem_2_v2.ingestion.extractor import TaskExtractor
from problem_2_v2.initialization.evaluator import CandidateEvaluation, CandidateEvaluatorAgent
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

    def test_run_with_native_websearch_retriever(self, tmp_path: Path) -> None:
        runner = SubprocessRunner(
            runs_dir=str(tmp_path / "runs"),
            timeout_seconds=5,
            python_executable=sys.executable,
        )
        debugger = DebuggerAgent(runner=runner, model="test", max_debug_rounds=1)
        extractor = TaskExtractor(use_llm=False)
        retriever = RetrieverAgent(model="test", num_candidates=2)
        evaluator = CandidateEvaluatorAgent(debugger=debugger, model="test")
        merger = ModelMergerAgent(debugger=debugger, model="test")
        pipeline = InitializationPipeline(
            extractor=extractor,
            retriever=retriever,
            evaluator=evaluator,
            merger=merger,
            use_baseline=False,
        )

        def _retriever_handler(msgs: object, info: object) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart(tool_name="final_result", args={"response": _CARD_ARGS})])

        with (
            retriever.agent.override(model=FunctionModel(_retriever_handler)),
            evaluator.agent.override(model=TestModel(custom_output_text=_FINAL_SCORE)),
            merger.agent.override(model=FunctionModel(_merger_model)),
        ):
            result = pipeline.run(_MD, dataset_dir="/data", run_id="websearch_e2e")

        assert isinstance(result, InitializationResult)
        assert len(result.candidates.candidates) == 2
        assert result.best_score == pytest.approx(0.55)


def _pipeline_with(
    tmp_path: Path,
    *,
    use_baseline: bool = False,
    baseline_path: str | None = None,
) -> InitializationPipeline:
    runner = SubprocessRunner(
        runs_dir=str(tmp_path / "runs"),
        timeout_seconds=5,
        python_executable=sys.executable,
    )
    debugger = DebuggerAgent(runner=runner, model="test", max_debug_rounds=1)
    return InitializationPipeline(
        extractor=TaskExtractor(use_llm=False),
        retriever=RetrieverAgent(
            provider=MockSearchProvider(
                results={
                    "ranking": [
                        SearchResult(title="t", url="https://e.com", snippet="s"),
                    ]
                }
            ),
            model="test",
            num_candidates=2,
        ),
        evaluator=CandidateEvaluatorAgent(debugger=debugger, model="test"),
        merger=ModelMergerAgent(debugger=debugger, model="test"),
        use_baseline=use_baseline,
        baseline_path=baseline_path,
    )


class TestBaselineSeeding:
    """Test official baseline starter code injection."""

    def test_baseline_card_injected_when_enabled(self, tmp_path: Path) -> None:
        baseline_file = tmp_path / "baseline.py"
        baseline_file.write_text("def baseline_main():\n    pass\n", encoding="utf-8")
        pipeline = _pipeline_with(tmp_path, use_baseline=True, baseline_path=str(baseline_file))
        with (
            pipeline.retriever.agent.override(model=TestModel(custom_output_args=_CARD_ARGS)),
            pipeline.evaluator.agent.override(model=TestModel(custom_output_text=_FINAL_SCORE)),
            pipeline.merger.agent.override(model=FunctionModel(function=_merger_model)),
        ):
            result = pipeline.run(_MD, dataset_dir="/data", run_id="base")
        assert result.candidates.candidates[0].model_name == "Official Baseline"
        assert "baseline_main" in result.candidates.candidates[0].example_code
        assert len(result.evaluations) == 3

    def test_baseline_off_by_default(self, tmp_path: Path) -> None:
        pipeline = _pipeline_with(tmp_path)
        with (
            pipeline.retriever.agent.override(model=TestModel(custom_output_args=_CARD_ARGS)),
            pipeline.evaluator.agent.override(model=TestModel(custom_output_text=_FINAL_SCORE)),
            pipeline.merger.agent.override(model=FunctionModel(function=_merger_model)),
        ):
            result = pipeline.run(_MD, dataset_dir="/data", run_id="base2")
        assert all(c.model_name != "Official Baseline" for c in result.candidates.candidates)

    def test_baseline_detected_from_workspace(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "baseline.py").write_text(
            "def workspace_baseline():\n    pass\n", encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        pipeline = _pipeline_with(tmp_path, use_baseline=True)
        with (
            pipeline.retriever.agent.override(model=TestModel(custom_output_args=_CARD_ARGS)),
            pipeline.evaluator.agent.override(model=TestModel(custom_output_text=_FINAL_SCORE)),
            pipeline.merger.agent.override(model=FunctionModel(function=_merger_model)),
        ):
            result = pipeline.run(_MD, dataset_dir="/data", run_id="base3")
        assert result.candidates.candidates[0].model_name == "Official Baseline"
        assert "workspace_baseline" in result.candidates.candidates[0].example_code


class TestMergerFallback:
    """Test the merger preserves the best individual on failed merges."""

    def test_merger_preserves_best_nonempty_individual(self, tmp_path: Path) -> None:
        runner = SubprocessRunner(
            runs_dir=str(tmp_path / "runs"),
            timeout_seconds=5,
            python_executable=sys.executable,
        )
        debugger = DebuggerAgent(runner=runner, model="test", max_debug_rounds=1)
        merger = ModelMergerAgent(debugger=debugger, model="test")
        spec = TaskSpecification.from_markdown(
            "**Task Type:** RECOMMENDER_RANKING\n"
            "**Metric Direction:** MAXIMIZE\n"
            "**Dataset Files:** train.csv\n",
            dataset_dir="/data",
        )
        empty_eval = CandidateEvaluation(
            model_name="Broken",
            code="",
            result=None,
            debug_rounds=0,
            score=None,
        )
        good_eval = CandidateEvaluation(
            model_name="Good",
            code="print('Final Validation Performance: 0.55')",
            result=None,
            debug_rounds=0,
            score=0.55,
        )
        with merger.agent.override(model=TestModel(custom_output_text="not python (:")):
            outcome = merger.merge(spec, [empty_eval, good_eval], run_id="m")
        assert outcome.final_code == good_eval.code
        assert outcome.final_score == pytest.approx(0.55)


class TestInitializationIterationLogging:
    """Test Stage 1 candidate and merge records in the unified iteration log."""

    def test_run_logs_candidate_and_merge_entries(
        self, pipeline: InitializationPipeline, tmp_path: Path
    ) -> None:
        with (
            pipeline.retriever.agent.override(model=TestModel(custom_output_args=_CARD_ARGS)),
            pipeline.evaluator.agent.override(model=TestModel(custom_output_text=_FINAL_SCORE)),
            pipeline.merger.agent.override(model=FunctionModel(function=_merger_model)),
        ):
            result = pipeline.run(_MD, dataset_dir="/data", run_id="ilog")

        log_path = Path(pipeline.merger.debugger.runner.runs_dir) / "ilog" / "iteration_logs.jsonl"
        assert log_path.is_file()
        records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        candidates = [r for r in records if r["target_component"] == "CANDIDATE_EVALUATION"]
        merges = [r for r in records if r["target_component"] == "MODEL_MERGER"]

        assert len(candidates) == len(result.evaluations)
        assert all(r["stage"] == "INITIALIZATION" for r in candidates)
        assert candidates[0]["iteration_id"] == "cand_1"
        assert all(r["success"] is True for r in candidates)
        assert all(r["validation_score"] == pytest.approx(0.55) for r in candidates)
        assert all(r["delta_from_baseline"] == pytest.approx(0.05) for r in candidates)
        assert all(r["metrics"] == {"primary": 0.55} for r in candidates)
        assert candidates[0]["code_diff"] != ""
        assert all(r["branch_index"] is None for r in candidates)

        assert len(merges) == len(result.outcome.steps)
        assert merges[0]["iteration_id"] == "merge_1"
        assert merges[0]["stage"] == "INITIALIZATION"
        assert merges[0]["success"] is True
        assert "Model2" in merges[0]["hypothesis"]
