"""Unit tests for the core enum types in `contracts.enums`."""

import pytest

from problem_2_v2.contracts.enums import (
    ComponentCategory,
    EnsembleMethod,
    MetricDirection,
    TaskType,
)


class TestMetricDirection:
    """Test `MetricDirection` comparison and delta helpers."""

    @pytest.mark.parametrize(
        ("direction", "score_a", "score_b", "expected"),
        [
            (MetricDirection.MAXIMIZE, 0.9, 0.8, True),
            (MetricDirection.MAXIMIZE, 0.8, 0.9, False),
            (MetricDirection.MAXIMIZE, 0.8, 0.8, False),
            (MetricDirection.MINIMIZE, 0.5, 0.9, True),
            (MetricDirection.MINIMIZE, 0.9, 0.5, False),
            (MetricDirection.MINIMIZE, 0.5, 0.5, False),
        ],
    )
    def test_is_better(
        self,
        direction: MetricDirection,
        score_a: float,
        score_b: float,
        expected: bool,
    ) -> None:
        assert direction.is_better(score_a, score_b) is expected

    def test_delta_is_positive_when_improving(self) -> None:
        """A positive delta always means improvement over baseline."""
        assert MetricDirection.MAXIMIZE.delta(0.95, 0.85) == pytest.approx(0.1)
        assert MetricDirection.MINIMIZE.delta(0.70, 0.90) == pytest.approx(0.2)

    def test_delta_is_negative_when_regressing(self) -> None:
        assert MetricDirection.MAXIMIZE.delta(0.80, 0.90) == pytest.approx(-0.1)
        assert MetricDirection.MINIMIZE.delta(0.95, 0.85) == pytest.approx(-0.1)

    def test_delta_zero_when_equal(self) -> None:
        assert MetricDirection.MAXIMIZE.delta(0.5, 0.5) == pytest.approx(0.0)
        assert MetricDirection.MINIMIZE.delta(0.5, 0.5) == pytest.approx(0.0)


class TestTaskType:
    """Test the `TaskType` enum members."""

    def test_members(self) -> None:
        assert TaskType.TABULAR_CLASSIFICATION.value == "TABULAR_CLASSIFICATION"
        assert TaskType.TABULAR_REGRESSION.value == "TABULAR_REGRESSION"
        assert TaskType.RECOMMENDER_RANKING.value == "RECOMMENDER_RANKING"
        assert TaskType.IMAGE_CLASSIFICATION.value == "IMAGE_CLASSIFICATION"
        assert TaskType.IMAGE_TO_IMAGE.value == "IMAGE_TO_IMAGE"
        assert TaskType.TEXT_CLASSIFICATION.value == "TEXT_CLASSIFICATION"
        assert TaskType.SEQ_TO_SEQ.value == "SEQ_TO_SEQ"
        assert TaskType.AUDIO_CLASSIFICATION.value == "AUDIO_CLASSIFICATION"
        assert TaskType.MULTIMODAL.value == "MULTIMODAL"

    def test_member_count(self) -> None:
        assert len(TaskType) == 9

    def test_accepts_exact_value_string(self) -> None:
        assert TaskType("TABULAR_REGRESSION") is TaskType.TABULAR_REGRESSION

    def test_rejects_unknown_value(self) -> None:
        with pytest.raises(ValueError):
            TaskType("TIME_SERIES_FORECASTING")


class TestComponentCategory:
    """Test the `ComponentCategory` enum members."""

    def test_members(self) -> None:
        assert ComponentCategory.DATA_PREPROCESSING.value == "DATA_PREPROCESSING"
        assert ComponentCategory.FEATURE_ENGINEERING.value == "FEATURE_ENGINEERING"
        assert ComponentCategory.MODEL_ARCHITECTURE.value == "MODEL_ARCHITECTURE"
        assert ComponentCategory.LOSS_AND_OPTIMIZER.value == "LOSS_AND_OPTIMIZER"
        assert ComponentCategory.HYPERPARAMETERS.value == "HYPERPARAMETERS"
        assert ComponentCategory.POST_PROCESSING.value == "POST_PROCESSING"

    def test_member_count(self) -> None:
        assert len(ComponentCategory) == 6


class TestEnsembleMethod:
    """Test the `EnsembleMethod` enum members."""

    def test_members(self) -> None:
        assert EnsembleMethod.SIMPLE_AVERAGE.value == "SIMPLE_AVERAGE"
        assert EnsembleMethod.WEIGHTED_AVERAGE.value == "WEIGHTED_AVERAGE"
        assert EnsembleMethod.STACKING_META_LEARNER.value == "STACKING_META_LEARNER"
        assert EnsembleMethod.RANK_AVERAGING.value == "RANK_AVERAGING"
        assert EnsembleMethod.BLENDING.value == "BLENDING"

    def test_member_count(self) -> None:
        assert len(EnsembleMethod) == 5
