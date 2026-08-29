"""LLM-driven ensembling subpackage.

Exposes the parallel candidate generation, the ensemble planner and
ensembler agents, and the iterative ensemble optimization pipeline
(Algorithm 3).
"""

from problem_2_v2.ensembling.ensembler import EnsemblerAgent, EnsembleRun
from problem_2_v2.ensembling.parallel import ParallelSolutionGenerator
from problem_2_v2.ensembling.pipeline import EnsemblePipeline, EnsembleResult
from problem_2_v2.ensembling.planner import EnsemblePlannerAgent

__all__ = [
    "EnsemblePipeline",
    "EnsemblePlannerAgent",
    "EnsembleResult",
    "EnsembleRun",
    "EnsemblerAgent",
    "ParallelSolutionGenerator",
]
