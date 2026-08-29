"""Integration tests for the unified contracts package API."""

import json
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel, ValidationError

from problem_2_v2.contracts import (
    AblationReport,
    AblationResultItem,
    AblationVariant,
    ComponentCategory,
    DataLeakageStatus,
    DataUsageStatus,
    EnsembleMethod,
    EnsembleStrategy,
    ExecutionResult,
    MetricDirection,
    ModelCard,
    PipelineArtifact,
    RefinementPlan,
    RetrievedCandidates,
    TargetCodeBlock,
    TaskSpecification,
    TaskType,
    compute_code_diff,
    extract_python_code,
    validate_python_syntax,
)

ALL_MODEL_CLASSES: list[type[BaseModel]] = [
    TaskSpecification,
    ExecutionResult,
    PipelineArtifact,
    ModelCard,
    RetrievedCandidates,
    AblationVariant,
    AblationResultItem,
    AblationReport,
    TargetCodeBlock,
    RefinementPlan,
    EnsembleStrategy,
    DataLeakageStatus,
    DataUsageStatus,
]


def sample_task_spec() -> TaskSpecification:
    return TaskSpecification.from_markdown(
        "**Task Name:** KuaiRand-Pure\n"
        "**Task Type:** RECOMMENDER_RANKING\n"
        "**Metric Name:** NDCG@10\n"
        "**Metric Direction:** MAXIMIZE\n"
        "**Target Variable:** is_click\n"
        "**Baseline Score:** 0.9123\n"
        "**Dataset Files:** train.csv\n"
        "**Description:** CTR prediction on short videos.\n",
        dataset_dir="/data/kuaipure",
    )


class TestPackageExports:
    """Test the unified contracts package surface."""

    def test_all_public_names_are_exported(self) -> None:
        assert TaskType.RECOMMENDER_RANKING is not None
        assert MetricDirection.MAXIMIZE is not None
        assert ComponentCategory.MODEL_ARCHITECTURE is not None
        assert EnsembleMethod.STACKING_META_LEARNER is not None
        assert callable(extract_python_code)
        assert callable(validate_python_syntax)
        assert callable(compute_code_diff)


class TestRoundTripSerialization:
    """Test JSON round-trip serialization across every contract model."""

    def test_every_model_round_trips(self) -> None:
        instances = [
            sample_task_spec(),
            ExecutionResult(
                success=True,
                stdout="Final Validation Performance: 0.87",
                stderr="",
                returncode=0,
                duration_seconds=10.0,
            ),
            PipelineArtifact(
                version=0,
                full_code="print('hi')",
                validation_score=0.5,
                parent_version=None,
                applied_diff=None,
                iteration_stage="baseline",
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
            ModelCard(
                model_name="LightGBM",
                rationale="fast",
                example_code="import lightgbm",
                library_dependencies=["lightgbm"],
            ),
            RetrievedCandidates(
                candidates=[
                    ModelCard(
                        model_name="XGBoost",
                        rationale="strong",
                        example_code="import xgboost",
                        library_dependencies=["xgboost"],
                    )
                ],
                query_used="ranking SOTA",
                total_found=1,
            ),
            AblationVariant(
                variant_id="v1",
                component_name="imputer",
                category=ComponentCategory.DATA_PREPROCESSING,
                hypothesis="median wins",
                modified_code_block="b",
                ablation_code="c",
            ),
            AblationResultItem(
                variant_id="v1",
                validation_score=0.95,
                delta_from_baseline=0.05,
                summary="helped",
            ),
            AblationReport(
                baseline_score=0.9,
                ablation_results=[
                    AblationResultItem(
                        variant_id="v1",
                        validation_score=0.95,
                        delta_from_baseline=0.05,
                        summary="helped",
                    )
                ],
                highest_impact_component="imputer",
                raw_log_summary="...",
            ),
            TargetCodeBlock(
                raw_code="x = 1",
                category=ComponentCategory.FEATURE_ENGINEERING,
                start_line=None,
                end_line=None,
                initial_plan="scale it",
            ),
            RefinementPlan(
                plan_id="p1",
                natural_language_plan="scale features",
                target_subcomponents=["scaler"],
                expected_gain="+0.01",
                iteration_index=1,
            ),
            EnsembleStrategy(
                method=EnsembleMethod.WEIGHTED_AVERAGE,
                natural_language_plan="weighted blend",
                meta_learner_type=None,
                candidate_solution_ids=["a", "b"],
                code_template=None,
            ),
            DataLeakageStatus(
                leakage_status="No Data Leakage",
                is_leaking=False,
                suspicious_code_block=None,
                corrected_code_block=None,
                explanation="clean",
            ),
            DataUsageStatus(
                all_data_used=True,
                missing_sources=[],
                usage_recommendations="ok",
                improved_code_block=None,
            ),
        ]
        for model in instances:
            restored = model.__class__.model_validate_json(model.model_dump_json())
            assert restored == model, f"round-trip failed for {model.__class__.__name__}"

    def test_dumped_json_is_plain_serializable(self) -> None:
        task = sample_task_spec()
        dumped = json.loads(task.model_dump_json())
        assert dumped["task_type"] == "RECOMMENDER_RANKING"
        assert dumped["metric_direction"] == "MAXIMIZE"
        assert dumped["subsample_size"] == 30000


class TestValidationSettings:
    """Test that all models enforce the strict config contract."""

    @pytest.mark.parametrize("model_cls", ALL_MODEL_CLASSES)
    def test_extra_fields_are_forbidden(self, model_cls: type[BaseModel]) -> None:
        assert model_cls.model_config.get("extra") == "forbid"

    @pytest.mark.parametrize("model_cls", ALL_MODEL_CLASSES)
    def test_assignment_is_validated(self, model_cls: type[BaseModel]) -> None:
        assert model_cls.model_config.get("validate_assignment") is True


class TestEndToEndWorkflow:
    """Test a realistic slice of the MLE-STAR contract workflow."""

    def test_full_pipeline_flow(self) -> None:
        spec = sample_task_spec()
        assert spec.metric_direction.is_better(0.95, spec.baseline_score)

        full = "def pipeline():\n    x = 1\n    return x\n"
        block = TargetCodeBlock(
            raw_code="x = 1",
            category=ComponentCategory.FEATURE_ENGINEERING,
            initial_plan="double it",
        )
        refined = block.replace_in(full, "x = 2")
        valid, error = validate_python_syntax(refined)
        assert valid
        assert error is None

        diff = compute_code_diff(full, refined)
        assert "+    x = 2" in diff

        result = ExecutionResult(
            success=True,
            stdout="Final Validation Performance: 0.93",
            stderr="",
            returncode=0,
            duration_seconds=5.0,
        )
        score = result.extract_validation_score(result.stdout)
        assert score is not None
        assert spec.metric_direction.delta(score, spec.baseline_score) == pytest.approx(0.0177)

        leakage = DataLeakageStatus(
            leakage_status="No Data Leakage",
            is_leaking=False,
            suspicious_code_block=None,
            corrected_code_block=None,
            explanation="clean",
        )
        assert leakage.is_leaking is False

        usage = DataUsageStatus(
            all_data_used=True,
            missing_sources=[],
            usage_recommendations="all good",
            improved_code_block=None,
        )
        assert usage.all_data_used

        strategy = EnsembleStrategy(
            method=EnsembleMethod.SIMPLE_AVERAGE,
            natural_language_plan="average probabilities",
            meta_learner_type=None,
            candidate_solution_ids=["s1", "s2"],
            code_template=None,
        )
        assert strategy.method is EnsembleMethod.SIMPLE_AVERAGE

        artifact = PipelineArtifact(
            version=1,
            full_code=refined,
            validation_score=score,
            parent_version=0,
            applied_diff=diff,
            iteration_stage="refine",
        )
        assert artifact.parent_version == 0
        assert artifact.applied_diff == diff

    def test_markdown_to_json_to_markdown_equivalent_spec(self) -> None:
        spec = sample_task_spec()
        restored = TaskSpecification.model_validate_json(spec.model_dump_json())
        assert restored == spec
        assert restored.metric_direction is MetricDirection.MAXIMIZE

    def test_invalid_inputs_raise_descriptive_validation_error(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ModelCard(
                model_name="",
                rationale="r",
                example_code="x = 1",
                library_dependencies=[],
            )
        assert "model_name" in str(exc_info.value)
