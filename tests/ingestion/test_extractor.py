"""Unit tests for the task ingestion extractor agent."""

import pytest
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.enums import MetricDirection, TaskType
from problem_2_v2.ingestion.extractor import TaskExtractor

_MD = (
    "**Task Name:** KuaiRand-Pure CTR Prediction\n"
    "**Task Type:** RECOMMENDER_RANKING\n"
    "**Metric Name:** NDCG@10\n"
    "**Metric Direction:** MAXIMIZE\n"
    "**Target Variable:** is_click\n"
    "**Baseline Score:** 0.9123\n"
    "**Dataset Files:** train.csv, test.csv\n"
    "**Description:** Predict user click behaviour on short videos.\n"
)

_LLM_SPEC_ARGS = {
    "task_name": "KuaiRand-Pure CTR Prediction",
    "task_type": "RECOMMENDER_RANKING",
    "metric_name": "NDCG@10",
    "metric_direction": "MAXIMIZE",
    "target_variable": "is_click",
    "baseline_score": 0.9123,
    "dataset_files": ["train.csv", "test.csv"],
    "description": "Predict user click behaviour on short videos.",
    "constraints": "",
}


class TestTaskExtractor:
    """Test the LLM-backed `TaskExtractor` and its heuristic fallback."""

    def test_extracts_spec_via_llm(self) -> None:
        extractor = TaskExtractor(model="test", use_llm=True)
        args = {**_LLM_SPEC_ARGS, "dataset_dir": "/data/input"}
        with extractor.agent.override(model=TestModel(custom_output_args=args)):
            spec = extractor.extract(_MD, dataset_dir="/data/input")
        assert spec.task_type is TaskType.RECOMMENDER_RANKING
        assert spec.metric_direction is MetricDirection.MAXIMIZE
        assert spec.baseline_score == pytest.approx(0.9123)

    def test_dataset_dir_is_always_authoritative(self) -> None:
        extractor = TaskExtractor(model="test", use_llm=True)
        args = {**_LLM_SPEC_ARGS, "dataset_dir": "/hallucinated/path"}
        with extractor.agent.override(model=TestModel(custom_output_args=args)):
            spec = extractor.extract(_MD, dataset_dir="/real/data")
        assert spec.dataset_dir == "/real/data"

    def test_falls_back_to_heuristic_when_llm_disabled(self) -> None:
        extractor = TaskExtractor(use_llm=False)
        spec = extractor.extract(_MD, dataset_dir="/data/input")
        assert spec.task_name == "KuaiRand-Pure CTR Prediction"
        assert spec.task_type is TaskType.RECOMMENDER_RANKING
        assert spec.metric_direction is MetricDirection.MAXIMIZE
        assert spec.subsample_size == 30000

    def test_falls_back_to_heuristic_when_llm_output_invalid(self) -> None:
        extractor = TaskExtractor(model="test", use_llm=True)
        with extractor.agent.override(
            model=TestModel(custom_output_args={"task_name": "missing everything else"})
        ):
            spec = extractor.extract(_MD, dataset_dir="/data/input")
        assert spec.task_type is TaskType.RECOMMENDER_RANKING
        assert spec.metric_direction is MetricDirection.MAXIMIZE

    def test_falls_back_to_heuristic_when_llm_raises(self) -> None:
        extractor = TaskExtractor(model="test", use_llm=True)

        def exploding_model(messages, info):
            raise RuntimeError("LLM backend down")

        from pydantic_ai.models.function import FunctionModel

        with extractor.agent.override(model=FunctionModel(function=exploding_model)):
            spec = extractor.extract(_MD, dataset_dir="/data/input")
        assert spec.task_type is TaskType.RECOMMENDER_RANKING
        assert spec.dataset_dir == "/data/input"

    def test_heuristic_parser_is_used_directly(self) -> None:
        extractor = TaskExtractor(use_llm=False)
        spec = extractor.extract(_MD, dataset_dir="/data/input")
        assert spec.dataset_files == ["train.csv", "test.csv"]
