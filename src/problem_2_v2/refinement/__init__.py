"""Targeted refinement subpackage.

Exposes the nested refinement pipeline (Algorithm 2) and its component
agents: ablation generation, summarization, code block extraction,
adaptive planning, and code refinement.
"""

from problem_2_v2.refinement.ablation import AblationAgent, AblationSummarizerAgent
from problem_2_v2.refinement.coder import CoderAgent
from problem_2_v2.refinement.extractor import CodeBlockExtractorAgent
from problem_2_v2.refinement.pipeline import RefinementPipeline, RefinementResult
from problem_2_v2.refinement.planner import RefinementPlannerAgent

__all__ = [
    "AblationAgent",
    "AblationSummarizerAgent",
    "CodeBlockExtractorAgent",
    "CoderAgent",
    "RefinementPipeline",
    "RefinementPlannerAgent",
    "RefinementResult",
]
