"""Unit tests for the retriever agent."""

from pydantic_ai.capabilities import WebSearch
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.enums import TaskType
from problem_2_v2.contracts.search import ModelCard
from problem_2_v2.contracts.task import TaskSpecification
from problem_2_v2.search.providers import MockSearchProvider, SearchResult
from problem_2_v2.search.retriever import RetrieverAgent

_JSON_CARDS = (
    '[{"model_name": "LightGBM", "rationale": "gradient boosting", '
    '"example_code": "import lightgbm as lgb\\nmodel = lgb.LGBMClassifier()", '
    '"library_dependencies": ["lightgbm"]}]'
)

_MARKDOWN_CARDS = (
    "- **LightGBM**: Gradient boosting for tabular data\n"
    "  ```python\n"
    "  import lightgbm as lgb\n"
    "  model = lgb.LGBMClassifier()\n"
    "  ```\n"
    "- **XGBoost**: Boosted trees\n"
    "  ```python\n"
    "  import xgboost as xgb\n"
    "  model = xgb.XGBClassifier()\n"
    "  ```\n"
)


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

    def test_init_defaults_to_websearch_capability(self) -> None:
        agent = RetrieverAgent(model="test")
        assert agent.provider is None
        assert agent.capabilities is not None
        assert any(isinstance(c, WebSearch) for c in agent.capabilities)

    def test_retrieve_with_dynamic_websearch_capability(self) -> None:
        agent = RetrieverAgent(model="test", num_candidates=2)
        args = [_card_args("LightGBM"), _card_args("DeepFM")]

        def _handler(msgs: object, info: object) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart(tool_name="final_result", args={"response": args})])

        with agent.agent.override(model=FunctionModel(_handler)):
            candidates = agent.retrieve(_spec())
        assert candidates.total_found == 2
        assert [c.model_name for c in candidates.candidates] == ["LightGBM", "DeepFM"]
        assert candidates.query_used == agent.build_query(_spec())

    def test_retrieve_queries_provider_when_explicitly_provided(self) -> None:
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


class TestDualModeParsing:
    """Test raw-text JSON/markdown parsing and domain fallbacks."""

    def test_parse_cards_from_raw_json_array(self) -> None:
        cards = RetrieverAgent._parse_cards(_JSON_CARDS)
        assert len(cards) == 1
        assert cards[0].model_name == "LightGBM"
        assert "lgb.LGBMClassifier" in cards[0].example_code

    def test_parse_cards_from_fenced_json(self) -> None:
        cards = RetrieverAgent._parse_cards(f"```json\n{_JSON_CARDS}\n```")
        assert len(cards) == 1
        assert cards[0].model_name == "LightGBM"

    def test_parse_cards_from_markdown_list(self) -> None:
        cards = RetrieverAgent._parse_cards(_MARKDOWN_CARDS)
        assert [c.model_name for c in cards] == ["LightGBM", "XGBoost"]
        assert "lgb.LGBMClassifier" in cards[0].example_code
        assert "xgb.XGBClassifier" in cards[1].example_code

    def test_parse_cards_returns_empty_for_garbage(self) -> None:
        assert RetrieverAgent._parse_cards("no useful structure here") == []

    def test_retrieve_falls_back_to_text_json_when_structured_empty(self) -> None:
        provider = MockSearchProvider(results={})
        agent = RetrieverAgent(provider=provider, model="test", num_candidates=2)
        with (
            agent.agent.override(model=TestModel(custom_output_args=[])),
            agent.text_agent.override(model=TestModel(custom_output_text=_JSON_CARDS)),
        ):
            candidates = agent.retrieve(_spec())
        assert [c.model_name for c in candidates.candidates] == ["LightGBM"]
        assert candidates.total_found == 1

    def test_retrieve_falls_back_to_domain_cards_when_empty(self) -> None:
        provider = MockSearchProvider(results={})
        agent = RetrieverAgent(provider=provider, model="test", num_candidates=4)
        with (
            agent.agent.override(model=TestModel(custom_output_args=[])),
            agent.text_agent.override(model=TestModel(custom_output_text="nothing usable")),
        ):
            candidates = agent.retrieve(_spec())
        assert len(candidates.candidates) > 0
        assert candidates.candidates[0].model_name == "Factorization Machine (BPR Loss)"
        assert candidates.total_found == len(candidates.candidates)

    def test_domain_fallback_cards_for_recommender(self) -> None:
        cards = RetrieverAgent.get_domain_fallback_cards(TaskType.RECOMMENDER_RANKING)
        names = [c.model_name for c in cards]
        assert "DeepFM" in names
        assert "LightGBM Ranker" in names
        assert "Item Popularity" in names

    def test_domain_fallback_cards_for_tabular(self) -> None:
        cards = RetrieverAgent.get_domain_fallback_cards(TaskType.TABULAR_CLASSIFICATION)
        names = [c.model_name for c in cards]
        assert {"LightGBM", "XGBoost", "CatBoost", "Multi-Layer Perceptron"} <= set(names)

    def test_domain_fallback_cards_never_empty(self) -> None:
        for task_type in TaskType:
            assert RetrieverAgent.get_domain_fallback_cards(task_type) != []

    def test_fallback_cards_are_valid_model_cards(self) -> None:
        for card in RetrieverAgent.get_domain_fallback_cards(TaskType.IMAGE_CLASSIFICATION):
            assert isinstance(card, ModelCard)
            assert card.model_name
            assert card.example_code
