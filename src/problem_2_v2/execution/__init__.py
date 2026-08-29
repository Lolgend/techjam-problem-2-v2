"""Unified execution environment and guardrail modules.

Exposes the reusable ``ExecutionGuardrailPipeline`` orchestrator and its
shared ``ExecutionConfig``. The ``FinalArtifactProducer``
($\\mathcal{A}_{\\text{finalizer}}$) is exported once implemented.
"""

from problem_2_v2.execution.pipeline import ExecutionConfig, ExecutionGuardrailPipeline

__all__ = [
    "ExecutionConfig",
    "ExecutionGuardrailPipeline",
]
