"""Unit tests for the retriever agent."""

from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.task import TaskSpecification
from problem_2_v2.search.providers import MockSearchProvider, SearchResult
from problem_2_v2.search.retriever import RetrieverAgent


def _spec() -> TaskSpecification:
    return TaskSpecification.from_markdown(
        "**Task Name:** KuaiRand-Pure\n"
        "**Task Type:** RECOMMENDER_RANKING\n"
        "**Metric Name:** NDCG@10\n"
        "**Metric Direction:** MAXIMIZE\n"
        "**Target Variable:** is_click\n"
        "**Baseline Score:** 0.91\n"
        "**Dataset Files:** train.csv\n"
        "**Description:** CTR prediction on short videos.\n",
        dataset_dir="/data",
    )


def _card_args(model_name: str) -> dict[str, object]:
    return {
        "model_name": model_name,
        "rationale": "state of the art for this task",
        "example_code": f"import {model_name.lower()}\nmodel = {model_name}()",
        "library_dependencies": [model_name.lower()],
    }


class TestRetrieverAgent:
    """Test query formulation, retrieval, and model card parsing."""

    def test_build_query_contains_task_and_metric(self) -> None:
        agent = RetrieverAgent(provider=MockSearchProvider(results={}))
        query = agent.build_query(_spec())
        assert "RECOMMENDER_RANKING" in query
        assert "NDCG@10" in query

    def test_retrieve_queries_provider_with_built_query(self) -> None:
        provider = MockSearchProvider(
            results={
                "ctr": [
                    SearchResult(title="t", url="https://e.com", snippet="s"),
                ]
            }
        )
        agent = RetrieverAgent(provider=provider, model="test", num_candidates=2)
        args = [_card_args("LightGBM"), _card_args("DeepFM")]
        with agent.agent.override(model=TestModel(custom_output_args=args)):
            candidates = agent.retrieve(_spec())
        assert candidates.query_used == agent.build_query(_spec())
        assert candidates.total_found == 2
        assert [c.model_name for c in candidates.candidates] == ["LightGBM", "DeepFM"]

    def test_retrieve_returns_requested_candidate_count(self) -> None:
        provider = MockSearchProvider(results={})
        agent = RetrieverAgent(provider=provider, model="test", num_candidates=4)
        args = [_card_args(f"Model{i}") for i in range(4)]
        with agent.agent.override(model=TestModel(custom_output_args=args)):
            candidates = agent.retrieve(_spec())
        assert len(candidates.candidates) == 4
        assert candidates.total_found == 4

    def test_retrieve_cleans_example_code_from_cards(self) -> None:
        provider = MockSearchProvider(results={})
        agent = RetrieverAgent(provider=provider, model="test", num_candidates=1)
        args = [
            {
                "model_name": "CatBoost",
                "rationale": "r",
                "example_code": "```python\nmodel = CatBoost()\n```",
                "library_dependencies": ["catboost"],
            }
        ]
        with agent.agent.override(model=TestModel(custom_output_args=args)):
            candidates = agent.retrieve(_spec())
        assert candidates.candidates[0].example_code == "model = CatBoost()"
        assert "```" not in candidates.candidates[0].example_code

    def test_retrieve_returns_empty_when_no_cards(self) -> None:
        provider = MockSearchProvider(results={})
        agent = RetrieverAgent(provider=provider, model="test", num_candidates=4)
        with agent.agent.override(model=TestModel(custom_output_args=[])):
            candidates = agent.retrieve(_spec())
        assert candidates.candidates == []
        assert candidates.total_found == 0
