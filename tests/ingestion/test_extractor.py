"""Unit tests for the task ingestion extractor agent."""

import pytest

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
    """Test deterministic `TaskExtractor` parsing and raw_description preservation."""

    def test_extract_returns_validated_spec(self) -> None:
        extractor = TaskExtractor()
        spec = extractor.extract(_MD, dataset_dir="/data/input")
        assert spec.task_name == "KuaiRand-Pure CTR Prediction"
        assert spec.task_type is TaskType.RECOMMENDER_RANKING
        assert spec.metric_name == "NDCG@10"
        assert spec.metric_direction is MetricDirection.MAXIMIZE
        assert spec.target_variable == "is_click"
        assert spec.baseline_score == pytest.approx(0.9123)
        assert spec.dataset_files == ["train.csv", "test.csv"]
        assert spec.dataset_dir == "/data/input"
        assert spec.raw_description == _MD.strip()

    def test_dataset_dir_is_always_authoritative(self) -> None:
        extractor = TaskExtractor()
        spec = extractor.extract(_MD, dataset_dir="/real/data")
        assert spec.dataset_dir == "/real/data"

    def test_handles_empty_markdown(self) -> None:
        extractor = TaskExtractor()
        spec = extractor.extract("**Task Type:** TABULAR_CLASSIFICATION\n", dataset_dir="/data")
        assert spec.task_type is TaskType.TABULAR_CLASSIFICATION
        assert spec.raw_description == "**Task Type:** TABULAR_CLASSIFICATION"
