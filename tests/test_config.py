"""Unit tests for ``MLEStarConfig`` and agent model_settings propagation."""

import pytest
from pydantic import ValidationError

from problem_2_v2.config import MLEStarConfig
from problem_2_v2.execution.finalizer import FinalArtifactProducer
from problem_2_v2.execution.pipeline import ExecutionGuardrailPipeline
from problem_2_v2.guardrails.leakage import DataLeakageCheckerAgent
from problem_2_v2.guardrails.usage import DataUsageCheckerAgent
from problem_2_v2.initialization.evaluator import CandidateEvaluatorAgent
from problem_2_v2.initialization.merger import ModelMergerAgent
from problem_2_v2.orchestrator import MLEStarPipeline
from problem_2_v2.refinement.ablation import AblationAgent, AblationSummarizerAgent
from problem_2_v2.refinement.coder import CoderAgent
from problem_2_v2.refinement.extractor import CodeBlockExtractorAgent
from problem_2_v2.refinement.planner import RefinementPlannerAgent
from problem_2_v2.runner.debugger import DebuggerAgent
from problem_2_v2.runner.sandbox import SubprocessRunner
from problem_2_v2.search.retriever import RetrieverAgent


class TestMLEStarConfig:
    """Test token limit and temperature settings on MLEStarConfig."""

    def test_default_config_has_none_model_settings(self) -> None:
        config = MLEStarConfig()
        assert config.max_tokens is None
        assert config.temperature is None
        assert config.get_model_settings() is None

    def test_configured_max_tokens_and_temperature(self) -> None:
        config = MLEStarConfig(max_tokens=4096, temperature=0.7)
        assert config.max_tokens == 4096
        assert config.temperature == 0.7
        settings = config.get_model_settings()
        assert settings == {"max_tokens": 4096, "temperature": 0.7}

    def test_configured_only_max_tokens(self) -> None:
        config = MLEStarConfig(max_tokens=2048)
        assert config.get_model_settings() == {"max_tokens": 2048}

    def test_configured_only_temperature(self) -> None:
        config = MLEStarConfig(temperature=0.2)
        assert config.get_model_settings() == {"temperature": 0.2}

    def test_invalid_max_tokens_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MLEStarConfig(max_tokens=0)

    def test_invalid_temperature_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MLEStarConfig(temperature=2.5)


class TestAgentModelSettingsPropagation:
    """Test that model_settings are properly passed to Pydantic AI Agent instances."""

    def test_retriever_agent_model_settings(self) -> None:
        settings = {"max_tokens": 1000}
        retriever = RetrieverAgent(model="openai:gpt-4o", model_settings=settings)
        assert retriever.agent.model_settings == settings
        assert retriever.text_agent.model_settings == settings

    def test_evaluator_agent_model_settings(self, tmp_path) -> None:
        runner = SubprocessRunner(runs_dir=str(tmp_path))
        debugger = DebuggerAgent(runner=runner, model="openai:gpt-4o")
        settings = {"max_tokens": 2000}
        evaluator = CandidateEvaluatorAgent(debugger=debugger, model_settings=settings)
        assert evaluator.agent.model_settings == settings

    def test_merger_agent_model_settings(self, tmp_path) -> None:
        runner = SubprocessRunner(runs_dir=str(tmp_path))
        debugger = DebuggerAgent(runner=runner, model="openai:gpt-4o")
        settings = {"max_tokens": 2000}
        merger = ModelMergerAgent(debugger=debugger, model_settings=settings)
        assert merger.agent.model_settings == settings

    def test_ablation_agents_model_settings(self, tmp_path) -> None:
        runner = SubprocessRunner(runs_dir=str(tmp_path))
        settings = {"max_tokens": 1500}
        ablation = AblationAgent(model_settings=settings)
        assert ablation.agent.model_settings == settings
        summarizer = AblationSummarizerAgent(runner=runner, model_settings=settings)
        assert summarizer.agent.model_settings == settings
        assert summarizer.debugger.agent.model_settings == settings

    def test_extractor_agent_model_settings(self) -> None:
        settings = {"max_tokens": 800}
        extractor = CodeBlockExtractorAgent(model_settings=settings)
        assert extractor.agent.model_settings == settings

    def test_planner_agent_model_settings(self) -> None:
        settings = {"max_tokens": 500}
        planner = RefinementPlannerAgent(model_settings=settings)
        assert planner.agent.model_settings == settings

    def test_coder_agent_model_settings(self) -> None:
        settings = {"max_tokens": 3000}
        coder = CoderAgent(model_settings=settings)
        assert coder.agent.model_settings == settings

    def test_guardrails_agents_model_settings(self) -> None:
        settings = {"max_tokens": 1200}
        leakage = DataLeakageCheckerAgent(model_settings=settings)
        assert leakage.check_agent.model_settings == settings
        assert leakage.repair_agent.model_settings == settings

        usage = DataUsageCheckerAgent(model_settings=settings)
        assert usage.agent.model_settings == settings

    def test_debugger_agent_model_settings(self, tmp_path) -> None:
        runner = SubprocessRunner(runs_dir=str(tmp_path))
        settings = {"max_tokens": 2500}
        debugger = DebuggerAgent(runner=runner, model_settings=settings)
        assert debugger.agent.model_settings == settings

    def test_finalizer_agent_model_settings(self) -> None:
        settings = {"max_tokens": 4000}
        finalizer = FinalArtifactProducer(model_settings=settings)
        assert finalizer.agent.model_settings == settings
        assert finalizer.debugger.agent.model_settings == settings

    def test_execution_guardrail_pipeline_model_settings(self, tmp_path) -> None:
        settings = {"max_tokens": 1000}
        pipeline = ExecutionGuardrailPipeline(
            runner=SubprocessRunner(runs_dir=str(tmp_path)),
            model_settings=settings,
        )
        assert pipeline.leakage.check_agent.model_settings == settings
        assert pipeline.usage.agent.model_settings == settings
        assert pipeline.debugger.agent.model_settings == settings

    def test_orchestrator_pipeline_wires_model_settings(self) -> None:
        config = MLEStarConfig(max_tokens=3500, temperature=0.3)
        pipeline = MLEStarPipeline(config=config)
        expected = {"max_tokens": 3500, "temperature": 0.3}
        assert pipeline.finalizer.agent.model_settings == expected
        assert pipeline.ensembler.agent.model_settings == expected
        assert pipeline.ensemble_pipeline.planner.agent.model_settings == expected
        assert pipeline.execution.leakage.check_agent.model_settings == expected

        init, refine = pipeline._build_branch(seed=42)
        assert init.retriever.agent.model_settings == expected
        assert init.evaluator.agent.model_settings == expected
        assert init.merger.agent.model_settings == expected
        assert refine.coder.agent.model_settings == expected
        assert refine.planner.agent.model_settings == expected
        assert refine.extractor.agent.model_settings == expected
        assert refine.ablation.agent.model_settings == expected
        assert refine.leakage.check_agent.model_settings == expected
        assert refine.usage.agent.model_settings == expected
