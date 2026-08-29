"""Unified execution environment and guardrail modules.

Exposes the reusable ``ExecutionGuardrailPipeline`` orchestrator, the
``FinalArtifactProducer`` ($\\mathcal{A}_{\\text{finalizer}}$), and their
shared ``ExecutionConfig``.
"""

from problem_2_v2.execution.finalizer import FinalArtifact, FinalArtifactProducer
from problem_2_v2.execution.pipeline import ExecutionConfig, ExecutionGuardrailPipeline

__all__ = [
    "ExecutionConfig",
    "ExecutionGuardrailPipeline",
    "FinalArtifact",
    "FinalArtifactProducer",
]
