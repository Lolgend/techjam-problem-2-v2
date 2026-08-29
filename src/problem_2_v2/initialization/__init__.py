"""Initialization subpackage.

Exposes the initial solution generation pipeline (Algorithm 1) together
with the candidate evaluator and model merger agents.
"""

from problem_2_v2.initialization.evaluator import CandidateEvaluation, CandidateEvaluatorAgent
from problem_2_v2.initialization.merger import MergeOutcome, ModelMergerAgent
from problem_2_v2.initialization.pipeline import InitializationPipeline, InitializationResult

__all__ = [
    "CandidateEvaluation",
    "CandidateEvaluatorAgent",
    "InitializationPipeline",
    "InitializationResult",
    "MergeOutcome",
    "ModelMergerAgent",
]
