"""Unit tests for ensembling and guardrail schemas."""

import pytest
from pydantic import ValidationError

from problem_2_v2.contracts.enums import EnsembleMethod
from problem_2_v2.contracts.guardrails import (
    DataLeakageStatus,
    DataUsageStatus,
    EnsembleStrategy,
)


class TestDataLeakageStatus:
    """Test the leakage detector and its paper-exact prompt normalization."""

    def test_parses_yes_data_leakage_response(self) -> None:
        status = DataLeakageStatus(
            leakage_status="Yes Data Leakage",
            is_leaking=False,
            suspicious_code_block="df = pd.concat([train, test])",
            corrected_code_block=None,
            explanation="Test data was concatenated into training.",
        )
        assert status.is_leaking is True
        assert status.leakage_status == "Yes Data Leakage"

    def test_parses_no_data_leakage_response(self) -> None:
        status = DataLeakageStatus(
            leakage_status="No Data Leakage",
            is_leaking=True,
            suspicious_code_block=None,
            corrected_code_block=None,
            explanation="Only train split is used.",
        )
        assert status.is_leaking is False

    def test_parses_case_insensitive_prompt_response(self) -> None:
        status = DataLeakageStatus(
            leakage_status="no data leakage",
            is_leaking=True,
            suspicious_code_block=None,
            corrected_code_block=None,
            explanation="Clean.",
        )
        assert status.is_leaking is False

    def test_structured_json_input_keeps_explicit_flag(self) -> None:
        status = DataLeakageStatus.model_validate(
            {
                "leakage_status": "CLEAN",
                "is_leaking": False,
                "suspicious_code_block": None,
                "corrected_code_block": None,
                "explanation": "Audited by static analysis.",
            }
        )
        assert status.is_leaking is False

    def test_auto_correction_is_optional(self) -> None:
        status = DataLeakageStatus(
            leakage_status="Yes Data Leakage",
            is_leaking=False,
            suspicious_code_block="leaky block",
            corrected_code_block="fixed block",
            explanation="Auto-corrected.",
        )
        assert status.corrected_code_block == "fixed block"

    def test_normalization_does_not_mutate_input_dict(self) -> None:
        payload = {
            "leakage_status": "Yes Data Leakage",
            "is_leaking": False,
            "suspicious_code_block": None,
            "corrected_code_block": None,
            "explanation": "x",
        }
        DataLeakageStatus.model_validate(payload)
        assert payload["is_leaking"] is False

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            DataLeakageStatus(
                leakage_status="No Data Leakage",
                is_leaking=False,
                suspicious_code_block=None,
                corrected_code_block=None,
                explanation="ok",
                extra="no",
            )

    def test_json_round_trip(self) -> None:
        status = DataLeakageStatus(
            leakage_status="No Data Leakage",
            is_leaking=False,
            suspicious_code_block=None,
            corrected_code_block=None,
            explanation="Clean.",
        )
        restored = DataLeakageStatus.model_validate_json(status.model_dump_json())
        assert restored == status
        assert restored.is_leaking is False


class TestDataUsageStatus:
    """Test the data ingestion auditor."""

    def test_instantiates(self) -> None:
        status = DataUsageStatus(
            all_data_used=False,
            missing_sources=["user_features.csv"],
            usage_recommendations="Join user features before training.",
            improved_code_block="pd.merge(train, user_features, on='user_id')",
        )
        assert status.all_data_used is False
        assert status.missing_sources == ["user_features.csv"]
        assert status.improved_code_block is not None

    def test_all_data_used(self) -> None:
        status = DataUsageStatus(
            all_data_used=True,
            missing_sources=[],
            usage_recommendations="All sources consumed.",
            improved_code_block=None,
        )
        assert status.all_data_used is True
        assert status.missing_sources == []

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            DataUsageStatus(
                all_data_used=True,
                missing_sources=[],
                usage_recommendations="",
                improved_code_block=None,
                stray=1,
            )

    def test_json_round_trip(self) -> None:
        status = DataUsageStatus(
            all_data_used=False,
            missing_sources=["a.csv"],
            usage_recommendations="use a.csv",
            improved_code_block="x = 1",
        )
        restored = DataUsageStatus.model_validate_json(status.model_dump_json())
        assert restored == status


class TestEnsembleStrategy:
    """Test the `EnsembleStrategy` planning model."""

    def test_instantiates_with_simple_method(self) -> None:
        strategy = EnsembleStrategy(
            method=EnsembleMethod.SIMPLE_AVERAGE,
            natural_language_plan="Average the softmax probabilities.",
            meta_learner_type=None,
            candidate_solution_ids=["sol_1", "sol_2"],
            code_template=None,
        )
        assert strategy.method is EnsembleMethod.SIMPLE_AVERAGE
        assert strategy.meta_learner_type is None
        assert strategy.candidate_solution_ids == ["sol_1", "sol_2"]

    def test_stacking_meta_learner(self) -> None:
        strategy = EnsembleStrategy(
            method=EnsembleMethod.STACKING_META_LEARNER,
            natural_language_plan="Train a logistic regression on OOF preds.",
            meta_learner_type="LogisticRegression",
            candidate_solution_ids=["sol_1"],
            code_template=None,
        )
        assert strategy.meta_learner_type == "LogisticRegression"

    def test_rejects_unknown_method(self) -> None:
        with pytest.raises(ValidationError):
            EnsembleStrategy(
                method="SUPER_ENSEMBLE",  # type: ignore[arg-type]
                natural_language_plan="p",
                meta_learner_type=None,
                candidate_solution_ids=[],
                code_template=None,
            )

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            EnsembleStrategy(
                method=EnsembleMethod.BLENDING,
                natural_language_plan="p",
                meta_learner_type=None,
                candidate_solution_ids=[],
                code_template=None,
                stray=1,
            )

    def test_json_round_trip(self) -> None:
        strategy = EnsembleStrategy(
            method=EnsembleMethod.WEIGHTED_AVERAGE,
            natural_language_plan="Grid-search weights.",
            meta_learner_type=None,
            candidate_solution_ids=["a", "b"],
            code_template="pred = w0*p0 + w1*p1",
        )
        restored = EnsembleStrategy.model_validate_json(strategy.model_dump_json())
        assert restored == strategy
        assert restored.method is EnsembleMethod.WEIGHTED_AVERAGE
