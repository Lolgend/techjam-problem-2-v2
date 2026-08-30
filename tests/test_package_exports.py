"""Package API unification tests.

Verifies that every subpackage exposes a clean ``__all__`` export list and
that the documented public symbols are importable from their package root.
"""

import importlib

import pytest

SUBMODULE_EXPORTS: dict[str, list[str]] = {
    "problem_2_v2.ingestion": ["TaskExtractor"],
    "problem_2_v2.search": [
        "RetrieverAgent",
        "SearchProvider",
        "SearchResult",
        "MockSearchProvider",
        "TavilySearchProvider",
        "GoogleSearchProvider",
        "DuckDuckGoSearchProvider",
    ],
    "problem_2_v2.initialization": [
        "InitializationPipeline",
        "InitializationResult",
        "CandidateEvaluatorAgent",
        "CandidateEvaluation",
        "ModelMergerAgent",
        "MergeOutcome",
    ],
    "problem_2_v2.refinement": [
        "RefinementPipeline",
        "RefinementResult",
        "AblationAgent",
        "AblationSummarizerAgent",
        "CodeBlockExtractorAgent",
        "RefinementPlannerAgent",
        "CoderAgent",
    ],
    "problem_2_v2.guardrails": [
        "DataLeakageCheckerAgent",
        "DataUsageCheckerAgent",
    ],
    "problem_2_v2.runner": [
        "SubprocessRunner",
        "DebuggerAgent",
        "DebugOutcome",
    ],
    "problem_2_v2.ensembling": [
        "ParallelSolutionGenerator",
        "EnsemblePlannerAgent",
        "EnsemblerAgent",
        "EnsemblePipeline",
        "EnsembleResult",
        "EnsembleRun",
    ],
    "problem_2_v2.execution": [
        "ExecutionGuardrailPipeline",
        "ExecutionConfig",
        "FinalArtifactProducer",
        "FinalArtifact",
    ],
    "problem_2_v2.contracts": [
        "TaskSpecification",
        "ExecutionResult",
        "PipelineArtifact",
        "TaskType",
        "MetricDirection",
        "ComponentCategory",
        "EnsembleMethod",
        "DataLeakageStatus",
        "DataUsageStatus",
        "EnsembleStrategy",
        "ModelCard",
        "RetrievedCandidates",
        "AblationReport",
        "AblationResultItem",
        "AblationVariant",
        "RefinementPlan",
        "TargetCodeBlock",
        "IterationLogEntry",
        "CentralIterationLogger",
        "compute_code_diff",
        "extract_python_code",
        "validate_python_syntax",
    ],
}


@pytest.mark.parametrize("module_name,expected", sorted(SUBMODULE_EXPORTS.items()))
def test_submodule_all_exports(module_name: str, expected: list[str]) -> None:
    module = importlib.import_module(module_name)
    assert hasattr(module, "__all__"), f"{module_name} is missing __all__"
    assert isinstance(module.__all__, list)
    assert len(module.__all__) > 0
    for name in expected:
        assert name in module.__all__, f"{name} missing from {module_name}.__all__"
        assert getattr(module, name, None) is not None, f"{module_name}.{name} not importable"


def test_all_exports_are_unique() -> None:
    for module_name in SUBMODULE_EXPORTS:
        module = importlib.import_module(module_name)
        assert len(module.__all__) == len(set(module.__all__)), f"duplicates in {module_name}"


def test_star_imports_resolve() -> None:
    for module_name in SUBMODULE_EXPORTS:
        namespace: dict[str, object] = {}
        exec(f"from {module_name} import *", namespace)  # noqa: S102
        exported = {name for name in namespace if not name.startswith("__")}
        assert exported == set(SUBMODULE_EXPORTS[module_name])
