"""Core enum types for the MLE-STAR contract layer.

All enums are plain ``str``-valued enums so they serialize cleanly to JSON
and can be matched against LLM-produced strings.
"""

from __future__ import annotations

from enum import Enum


class TaskType(str, Enum):
    """The family of machine learning tasks MLE-STAR can tackle."""

    TABULAR_CLASSIFICATION = "TABULAR_CLASSIFICATION"
    TABULAR_REGRESSION = "TABULAR_REGRESSION"
    RECOMMENDER_RANKING = "RECOMMENDER_RANKING"
    IMAGE_CLASSIFICATION = "IMAGE_CLASSIFICATION"
    IMAGE_TO_IMAGE = "IMAGE_TO_IMAGE"
    TEXT_CLASSIFICATION = "TEXT_CLASSIFICATION"
    SEQ_TO_SEQ = "SEQ_TO_SEQ"
    AUDIO_CLASSIFICATION = "AUDIO_CLASSIFICATION"
    MULTIMODAL = "MULTIMODAL"


class MetricDirection(str, Enum):
    """Direction of the evaluation metric: higher is better or lower is better.

    ``MAXIMIZE`` applies to metrics like NDCG@K, Recall@K, Accuracy, AUROC
    and R2. ``MINIMIZE`` applies to metrics like RMSE, LogLoss, MAE and
    RMLSE.
    """

    MAXIMIZE = "MAXIMIZE"
    MINIMIZE = "MINIMIZE"

    def is_better(self, score_a: float, score_b: float) -> bool:
        """Return whether ``score_a`` is better than ``score_b``.

        Args:
            score_a: The candidate score.
            score_b: The score to compare against.

        Returns:
            True when ``score_a`` outperforms ``score_b`` under this
            direction; equality is never "better".
        """
        if self is MetricDirection.MAXIMIZE:
            return score_a > score_b
        return score_a < score_b

    def delta(self, score: float, baseline: float) -> float:
        """Compute the signed improvement of ``score`` over ``baseline``.

        Args:
            score: The agent's validation score.
            baseline: The baseline score to compare against.

        Returns:
            A signed delta that is positive when the agent improved over
            the baseline, negative when it regressed, and zero on equality.
            For ``MAXIMIZE`` this is ``score - baseline``; for ``MINIMIZE``
            it is ``baseline - score`` so that positive always means
            improvement.
        """
        if self is MetricDirection.MAXIMIZE:
            return score - baseline
        return baseline - score


class ComponentCategory(str, Enum):
    """Pipeline component categories targeted for ablation and refinement."""

    DATA_PREPROCESSING = "DATA_PREPROCESSING"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    MODEL_ARCHITECTURE = "MODEL_ARCHITECTURE"
    LOSS_AND_OPTIMIZER = "LOSS_AND_OPTIMIZER"
    HYPERPARAMETERS = "HYPERPARAMETERS"
    POST_PROCESSING = "POST_PROCESSING"


class EnsembleMethod(str, Enum):
    """LLM-proposed ensembling strategies across candidate solutions."""

    SIMPLE_AVERAGE = "SIMPLE_AVERAGE"
    WEIGHTED_AVERAGE = "WEIGHTED_AVERAGE"
    STACKING_META_LEARNER = "STACKING_META_LEARNER"
    RANK_AVERAGING = "RANK_AVERAGING"
    BLENDING = "BLENDING"
