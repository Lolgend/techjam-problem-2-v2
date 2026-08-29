"""Unified public API for the MLE-STAR contract layer.

This package exposes every data contract, enum, and code utility used by
the MLE-STAR agent framework under a single import surface.
"""

from problem_2_v2.contracts.code_utils import (
    compute_code_diff,
    extract_python_code,
    validate_python_syntax,
)
from problem_2_v2.contracts.enums import (
    ComponentCategory,
    EnsembleMethod,
    MetricDirection,
    TaskType,
)
from problem_2_v2.contracts.guardrails import (
    DataLeakageStatus,
    DataUsageStatus,
    EnsembleStrategy,
)
from problem_2_v2.contracts.refinement import (
    AblationReport,
    AblationResultItem,
    AblationVariant,
    RefinementPlan,
    TargetCodeBlock,
)
from problem_2_v2.contracts.search import ModelCard, RetrievedCandidates
from problem_2_v2.contracts.task import (
    ExecutionResult,
    PipelineArtifact,
    TaskSpecification,
)

__all__ = [
    "AblationReport",
    "AblationResultItem",
    "AblationVariant",
    "ComponentCategory",
    "DataLeakageStatus",
    "DataUsageStatus",
    "EnsembleMethod",
    "EnsembleStrategy",
    "ExecutionResult",
    "MetricDirection",
    "ModelCard",
    "PipelineArtifact",
    "RefinementPlan",
    "RetrievedCandidates",
    "TargetCodeBlock",
    "TaskSpecification",
    "TaskType",
    "compute_code_diff",
    "extract_python_code",
    "validate_python_syntax",
]
