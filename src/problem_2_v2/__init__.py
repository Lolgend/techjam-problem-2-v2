"""MLE-STAR agent framework: public API surface.

Exports the master orchestrator, its configuration, the structured run
result, and the CLI entry point.
"""

from problem_2_v2.cli import main
from problem_2_v2.config import MLEStarConfig
from problem_2_v2.orchestrator import MLEStarPipeline, MLEStarResult

__all__ = ["MLEStarConfig", "MLEStarPipeline", "MLEStarResult", "main"]
