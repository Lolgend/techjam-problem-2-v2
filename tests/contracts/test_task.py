"""Unit tests for task specification, execution telemetry, and artifact lineage."""

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from problem_2_v2.contracts.enums import MetricDirection, TaskType
from problem_2_v2.contracts.task import (
    ExecutionResult,
    PipelineArtifact,
    TaskSpecification,
)


class TestTaskSpecification:
    """Test `TaskSpecification` markdown parsing and validation."""

    def test_from_markdown_parses_full_description(self) -> None:
        md_text = (
            "**Task Name:** KuaiRand-Pure CTR Prediction\n"
            "**Task Type:** RECOMMENDER_RANKING\n"
            "**Metric Name:** NDCG@10\n"
            "**Metric Direction:** MAXIMIZE\n"
            "**Target Variable:** is_click\n"
            "**Baseline Score:** 0.9123\n"
            "**Dataset Files:** train.csv, test.csv\n"
            "**Description:** Predict user click behaviour on short videos.\n"
            "This is a multi-task ranking benchmark.\n"
            "**Constraints:** Must not use test labels.\n"
            "GPU memory limited to 24GB.\n"
        )
        spec = TaskSpecification.from_markdown(md_text, dataset_dir="/data/kuaipure")

        assert spec.task_name == "KuaiRand-Pure CTR Prediction"
        assert spec.task_type is TaskType.RECOMMENDER_RANKING
        assert spec.metric_name == "NDCG@10"
        assert spec.metric_direction is MetricDirection.MAXIMIZE
        assert spec.target_variable == "is_click"
        assert spec.baseline_score == pytest.approx(0.9123)
        assert spec.dataset_dir == "/data/kuaipure"
        assert spec.dataset_files == ["train.csv", "test.csv"]
        assert "Predict user click behaviour on short videos." in spec.description
        assert "multi-task ranking benchmark." in spec.description
        assert "Must not use test labels." in spec.constraints
        assert spec.subsample_size == 30000

    def test_from_markdown_uses_default_subsample_size(self) -> None:
        md_text = "**Task Type:** TABULAR_CLASSIFICATION\n"
        spec = TaskSpecification.from_markdown(md_text, dataset_dir="/data")
        assert spec.subsample_size == 30000

    def test_from_markdown_accepts_custom_subsample_size(self) -> None:
        md_text = "**Task Type:** TABULAR_REGRESSION\n**Subsample Size:** 5000\n"
        spec = TaskSpecification.from_markdown(md_text, dataset_dir="/data")
        assert spec.subsample_size == 5000

    def test_from_markdown_handles_bold_heading_labels(self) -> None:
        md_text = "### **Task Name:** Audio Tagging\n## **Task Type:** AUDIO_CLASSIFICATION\n"
        spec = TaskSpecification.from_markdown(md_text, dataset_dir="/data")
        assert spec.task_name == "Audio Tagging"
        assert spec.task_type is TaskType.AUDIO_CLASSIFICATION

    def test_from_markdown_rejects_unknown_task_type(self) -> None:
        md_text = "**Task Type:** DEEP_LEARNING_MAGIC\n"
        with pytest.raises(ValidationError):
            TaskSpecification.from_markdown(md_text, dataset_dir="/data")

    def test_from_markdown_rejects_unknown_metric_direction(self) -> None:
        md_text = "**Task Type:** TABULAR_REGRESSION\n**Metric Direction:** SIDEWAYS\n"
        with pytest.raises(ValidationError):
            TaskSpecification.from_markdown(md_text, dataset_dir="/data")

    def test_programmatic_instantiation(self) -> None:
        spec = TaskSpecification(
            task_name="demo",
            task_type=TaskType.TABULAR_CLASSIFICATION,
            description="classify things",
            metric_name="AUROC",
            metric_direction=MetricDirection.MAXIMIZE,
            target_variable="label",
            dataset_dir="/data",
            dataset_files=["train.csv"],
            baseline_score=0.7,
            constraints="none",
        )
        assert spec.task_name == "demo"
        assert spec.metric_direction is MetricDirection.MAXIMIZE

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            TaskSpecification(
                task_name="demo",
                task_type=TaskType.TABULAR_CLASSIFICATION,
                description="d",
                metric_name="AUROC",
                metric_direction=MetricDirection.MAXIMIZE,
                target_variable="label",
                dataset_dir="/data",
                dataset_files=["train.csv"],
                baseline_score=0.7,
                constraints="none",
                sneaky_field="nope",
            )

    def test_assignment_is_validated(self) -> None:
        spec = TaskSpecification(
            task_name="demo",
            task_type=TaskType.TABULAR_CLASSIFICATION,
            description="d",
            metric_name="AUROC",
            metric_direction=MetricDirection.MAXIMIZE,
            target_variable="label",
            dataset_dir="/data",
            dataset_files=["train.csv"],
            baseline_score=0.7,
            constraints="none",
        )
        with pytest.raises(ValidationError):
            spec.task_type = "NOT_A_TASK_TYPE"  # type: ignore[assignment]

    def test_json_round_trip(self) -> None:
        spec = TaskSpecification(
            task_name="demo",
            task_type=TaskType.TABULAR_REGRESSION,
            description="regress things",
            metric_name="RMSE",
            metric_direction=MetricDirection.MINIMIZE,
            target_variable="price",
            dataset_dir="/data",
            dataset_files=["train.csv", "test.csv"],
            baseline_score=1.2,
            constraints="none",
        )
        restored = TaskSpecification.model_validate_json(spec.model_dump_json())
        assert restored == spec


class TestExecutionResult:
    """Test `ExecutionResult` subprocess telemetry and score parsing."""

    def test_instantiates_with_valid_fields(self) -> None:
        result = ExecutionResult(
            success=True,
            stdout="Final Validation Performance: 0.9123",
            stderr="",
            returncode=0,
            duration_seconds=12.5,
        )
        assert result.success is True
        assert result.returncode == 0
        assert result.duration_seconds == pytest.approx(12.5)
        assert result.validation_score is None

    def test_extracts_validation_score_from_stdout(self) -> None:
        stdout = "Training complete in 42s\nFinal Validation Performance: 0.8731\nAll done."
        result = ExecutionResult(
            success=True,
            stdout=stdout,
            stderr="",
            returncode=0,
            duration_seconds=42.0,
        )
        assert result.validation_score is None
        score = result.extract_validation_score(stdout)
        assert score == pytest.approx(0.8731)

    def test_parse_returns_none_when_no_score_present(self) -> None:
        result = ExecutionResult(
            success=False,
            stdout="Something crashed.",
            stderr="Traceback (most recent call last):",
            returncode=1,
            duration_seconds=3.1,
        )
        assert result.extract_validation_score("Something crashed.") is None

    def test_parse_handles_scientific_notation(self) -> None:
        result = ExecutionResult(
            success=True,
            stdout="Final Validation Performance: 1.5e-05",
            stderr="",
            returncode=0,
            duration_seconds=1.0,
        )
        assert result.extract_validation_score(
            "Final Validation Performance: 1.5e-05"
        ) == pytest.approx(1.5e-05)

    def test_optional_fields_are_settable(self) -> None:
        result = ExecutionResult(
            success=False,
            stdout="",
            stderr="boom",
            returncode=1,
            duration_seconds=0.5,
            validation_score=None,
            error_traceback="Traceback: boom",
            gpu_memory_mb=1024.0,
        )
        assert result.error_traceback == "Traceback: boom"
        assert result.gpu_memory_mb == pytest.approx(1024.0)

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionResult(
                success=True,
                stdout="",
                stderr="",
                returncode=0,
                duration_seconds=1.0,
                unexpected="x",
            )

    def test_json_round_trip(self) -> None:
        result = ExecutionResult(
            success=True,
            stdout="Final Validation Performance: 0.5",
            stderr="",
            returncode=0,
            duration_seconds=9.9,
            validation_score=0.5,
        )
        restored = ExecutionResult.model_validate_json(result.model_dump_json())
        assert restored == result


class TestPipelineArtifact:
    """Test `PipelineArtifact` version lineage and serialization."""

    def test_instantiates(self) -> None:
        artifact = PipelineArtifact(
            version=0,
            full_code="print('baseline')",
            validation_score=0.5,
            parent_version=None,
            applied_diff=None,
            iteration_stage="baseline",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        assert artifact.version == 0
        assert artifact.parent_version is None
        assert artifact.iteration_stage == "baseline"

    def test_lineage_links_child_to_parent(self) -> None:
        parent = PipelineArtifact(
            version=0,
            full_code="x = 1",
            validation_score=0.5,
            iteration_stage="baseline",
        )
        child = PipelineArtifact(
            version=1,
            full_code="x = 2",
            validation_score=0.6,
            parent_version=parent.version,
            applied_diff="-x = 1\n+x = 2",
            iteration_stage="refine",
        )
        assert child.parent_version == 0
        assert child.applied_diff == "-x = 1\n+x = 2"

    def test_json_round_trip_preserves_timestamp(self) -> None:
        artifact = PipelineArtifact(
            version=2,
            full_code="print('hi')",
            validation_score=0.7,
            parent_version=1,
            applied_diff="+print('hi')",
            iteration_stage="refine",
            timestamp=datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc),
        )
        dumped = json.loads(artifact.model_dump_json())
        assert dumped["version"] == 2
        assert dumped["parent_version"] == 1
        restored = PipelineArtifact.model_validate_json(artifact.model_dump_json())
        assert restored == artifact

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            PipelineArtifact(
                version=0,
                full_code="x",
                validation_score=0.1,
                parent_version=None,
                applied_diff=None,
                iteration_stage="baseline",
                extra="no",
            )
