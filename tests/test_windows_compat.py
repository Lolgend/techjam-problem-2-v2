"""Windows AsyncIO compatibility and thread-safe search provider tests.

Verifies the ``WindowsSelectorEventLoopPolicy`` is configured on Windows to
prevent the Proactor ``WinError 10038`` socket teardown crash, that
``DuckDuckGoSearchProvider`` serializes concurrent searches, degrades
gracefully on network/socket errors, and scopes fresh backend sessions,
and that concurrent parallel branches execute without WinError 10038.
"""

import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.task import TaskSpecification
from problem_2_v2.ensembling.parallel import ParallelSolutionGenerator
from problem_2_v2.guardrails.leakage import DataLeakageCheckerAgent
from problem_2_v2.guardrails.usage import DataUsageCheckerAgent
from problem_2_v2.ingestion.extractor import TaskExtractor
from problem_2_v2.initialization.evaluator import CandidateEvaluatorAgent
from problem_2_v2.initialization.merger import ModelMergerAgent
from problem_2_v2.initialization.pipeline import InitializationPipeline
from problem_2_v2.orchestrator import configure_event_loop_policy
from problem_2_v2.refinement.ablation import AblationAgent, AblationSummarizerAgent
from problem_2_v2.refinement.coder import CoderAgent
from problem_2_v2.refinement.extractor import CodeBlockExtractorAgent
from problem_2_v2.refinement.pipeline import RefinementPipeline
from problem_2_v2.refinement.planner import RefinementPlannerAgent
from problem_2_v2.runner.debugger import DebuggerAgent
from problem_2_v2.runner.sandbox import SubprocessRunner
from problem_2_v2.search.providers import DuckDuckGoSearchProvider, MockSearchProvider, SearchResult
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


def _branch_factory(tmp_path: Path):
    """TestModel branch factory producing concurrent subprocess solutions."""

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


def _spec() -> TaskSpecification:
    return TaskSpecification.from_markdown(_MD, dataset_dir="/data")


class TestWindowsEventLoopPolicy:
    """Test the Windows selector event loop policy configuration."""

    def test_selector_policy_configured_on_windows(self) -> None:
        previous = asyncio.get_event_loop_policy()
        try:
            configure_event_loop_policy()
            policy = asyncio.get_event_loop_policy()
            if sys.platform == "win32":
                assert isinstance(policy, asyncio.WindowsSelectorEventLoopPolicy)
        finally:
            asyncio.set_event_loop_policy(previous)

    def test_policy_configuration_is_idempotent(self) -> None:
        previous = asyncio.get_event_loop_policy()
        try:
            configure_event_loop_policy()
            configure_event_loop_policy()
        finally:
            asyncio.set_event_loop_policy(previous)


class TestThreadSafeSearchProvider:
    """Test concurrent DuckDuckGo searches and graceful degradation."""

    def test_concurrent_searches_serialize_safely(self) -> None:
        class FakeDDGS:
            def __init__(self) -> None:
                self.calls = 0

            def text(self, keywords, max_results=None, **_):
                current = self.calls
                time.sleep(0.001)
                self.calls = current + 1
                return [{"title": f"Hit {self.calls}", "href": "https://e.com", "body": "s"}]

        backend = FakeDDGS()
        provider = DuckDuckGoSearchProvider(backend=backend)

        def run_search(_: int) -> list[object]:
            return provider.search("recommendation model")

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(run_search, range(16)))
        assert len(results) == 16
        assert all(len(r) == 1 for r in results)
        assert backend.calls == 16

    def test_search_error_degrades_to_empty(self) -> None:
        class BrokenDDGS:
            def text(self, keywords, max_results=None, **_):
                raise OSError("An operation was attempted on something that is not a socket")

        provider = DuckDuckGoSearchProvider(backend=BrokenDDGS())
        assert provider.search("query") == []

    def test_scoped_backend_session_per_call(self, monkeypatch) -> None:
        instances = {"count": 0}

        class FakeDDGS:
            def __init__(self, timeout: int = 20) -> None:
                instances["count"] += 1

            def text(self, keywords, max_results=None, **_):
                return [{"title": "t", "href": "https://e.com", "body": "b"}]

        monkeypatch.setattr("problem_2_v2.search.providers.DDGS", FakeDDGS)
        provider = DuckDuckGoSearchProvider()
        assert len(provider.search("q")) == 1
        assert len(provider.search("q")) == 1
        assert instances["count"] == 2


class TestConcurrentBranchExecution:
    """Test concurrent parallel branches run without WinError 10038."""

    def test_parallel_branches_execute_without_winerror(self, tmp_path: Path) -> None:
        previous = asyncio.get_event_loop_policy()
        try:
            configure_event_loop_policy()
            data_dir = tmp_path / "data"
            data_dir.mkdir(exist_ok=True)
            (data_dir / "train.csv").write_text("x,y\n1,0\n", encoding="utf-8")
            generator = ParallelSolutionGenerator(
                branch_builder=_branch_factory(tmp_path), num_branches=2
            )

            async def run() -> list[object]:
                return await generator.generate(
                    _MD, dataset_dir=str(data_dir), run_id="win", seeds=[0, 1]
                )

            artifacts = asyncio.run(run())
        finally:
            asyncio.set_event_loop_policy(previous)

        assert len(artifacts) == 2
        assert [a.validation_score for a in artifacts] == [0.51, 0.52]
