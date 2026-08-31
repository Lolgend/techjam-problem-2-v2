"""Unit tests for the ablation generation and summarization agents."""

import sys
from pathlib import Path

import pytest
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.refinement import AblationReport
from problem_2_v2.refinement.ablation import AblationAgent, AblationSummarizerAgent
from problem_2_v2.runner.debugger import DebuggerAgent
from problem_2_v2.runner.sandbox import SubprocessRunner

SOLUTION = (
    "import pandas as pd\n"
    "train = pd.read_csv('./input/train.csv')\n"
    "X = train.drop(columns=['label'])\n"
    "y = train['label']\n"
    "model = LogisticRegression()\n"
    "model.fit(X, y)\n"
    "print('Final Validation Performance: 0.80')\n"
)

ABLATION_CODE = (
    "from sklearn.linear_model import LogisticRegression\n"
    "for strategy in ['median', 'mean']:\n"
    "    print(f'{strategy} imputation: 0.75')"
)


class TestAblationAgent:
    """Test Figure 12 ablation script generation."""

    def test_generates_ablation_script(self) -> None:
        agent = AblationAgent(model="test")
        with agent.agent.override(
            model=TestModel(custom_output_text=f"```python\n{ABLATION_CODE}\n```")
        ):
            code = agent.generate_ablation(SOLUTION, previous_ablations=[])
        assert code == ABLATION_CODE
        assert "```" not in code

    def test_prompt_includes_solution_and_history(self) -> None:
        agent = AblationAgent(model="test")
        captured: dict[str, str] = {}

        def capturing_model(messages, info):
            captured["prompt"] = messages[-1].parts[0].content
            return ModelResponse(parts=[TextPart(content=ABLATION_CODE)])

        with agent.agent.override(model=FunctionModel(function=capturing_model)):
            agent.generate_ablation(
                SOLUTION,
                previous_ablations=["Feature scaling had the biggest impact."],
            )
        assert "Python solution" in captured["prompt"]
        assert "Feature scaling had the biggest impact." in captured["prompt"]
        assert "2-3" in captured["prompt"] or "2 or 3" in captured["prompt"]

    def test_build_prompt_format(self) -> None:
        prompt = AblationAgent.build_prompt(
            SOLUTION,
            previous_ablations=["Feature scaling: +0.02", "Tree depth: -0.01"],
        )
        assert "# Introduction" in prompt
        assert "- You are a Kaggle grandmaster attending a competition." in prompt
        assert "perform an ablation study on the current" in prompt
        assert "# Python solution" in prompt
        assert SOLUTION in prompt
        assert "## Previous ablation study result {0}" in prompt
        assert "Feature scaling: +0.02" in prompt
        assert "## Previous ablation study result {1}" in prompt
        assert "Tree depth: -0.01" in prompt
        assert "# Instructions" in prompt
        assert "generate a simple Python code that performs an ablation study" in prompt
        assert "modifying or disabling parts (2-3 parts)" in prompt
        assert "# Response format" in prompt
        assert "The Python code for the ablation study should not load test data." in prompt
        assert "contributes the most to the" in prompt


class TestAblationSummarizerAgent:
    """Test Figure 13 raw-log summarization into `AblationReport`."""

    @pytest.fixture()
    def runner(self, tmp_path: Path) -> SubprocessRunner:
        return SubprocessRunner(
            runs_dir=str(tmp_path / "runs"),
            timeout_seconds=5,
            python_executable=sys.executable,
        )

    @pytest.fixture()
    def summarizer(self, runner: SubprocessRunner) -> AblationSummarizerAgent:
        return AblationSummarizerAgent(runner=runner, model="test")

    def _report_args(self) -> dict[str, object]:
        return {
            "baseline_score": 0.80,
            "ablation_results": [
                {
                    "variant_id": "imputer_median",
                    "validation_score": 0.82,
                    "delta_from_baseline": 0.02,
                    "summary": "Median imputation improved score.",
                },
                {
                    "variant_id": "imputer_mean",
                    "validation_score": 0.79,
                    "delta_from_baseline": -0.01,
                    "summary": "Mean imputation hurt score.",
                },
            ],
            "highest_impact_component": "imputer",
            "raw_log_summary": "...",
        }

    def test_executes_and_summarizes_into_report(self, summarizer: AblationSummarizerAgent) -> None:
        with summarizer.agent.override(model=TestModel(custom_output_args=self._report_args())):
            report = summarizer.summarize(ABLATION_CODE, run_id="r")
        assert isinstance(report, AblationReport)
        assert report.highest_impact_component == "imputer"
        assert len(report.ablation_results) == 2
        top = report.highest_impact_result()
        assert top is not None
        assert top.variant_id == "imputer_median"

    def test_falls_back_to_heuristic_parse_when_llm_fails(
        self, summarizer: AblationSummarizerAgent
    ) -> None:
        def exploding_model(messages, info):
            raise RuntimeError("LLM backend down")

        code = (
            "print('Final Validation Performance: 0.80')\n"
            "print('variant A: 0.85')\n"
            "print('variant B: 0.78')\n"
        )
        with summarizer.agent.override(model=FunctionModel(function=exploding_model)):
            report = summarizer.summarize(code, run_id="r")
        assert report.raw_log_summary != ""
        assert report.highest_impact_component in ("variant A", "variant B", "variant_a")

    def test_prompt_contains_code_and_output(self, summarizer: AblationSummarizerAgent) -> None:
        captured: dict[str, str] = {}

        def capturing_model(messages, info):
            captured["prompt"] = messages[-1].parts[0].content
            report_json = (
                '{"baseline_score": 0.0, "ablation_results": [], '
                '"highest_impact_component": "x", "raw_log_summary": "y"}'
            )
            return ModelResponse(parts=[TextPart(content=report_json)])

        with summarizer.agent.override(model=FunctionModel(function=capturing_model)):
            summarizer.summarize(ABLATION_CODE, run_id="r")
        assert ABLATION_CODE in captured["prompt"]
        assert "Ablation study results" in captured["prompt"]

    def test_summarizer_build_prompt_format(self) -> None:
        prompt = AblationSummarizerAgent.build_prompt(
            ablation_code=ABLATION_CODE,
            raw_output="variant A: 0.85\nvariant B: 0.78",
        )
        assert "# Your code for ablation study was:" in prompt
        assert ABLATION_CODE in prompt
        assert "# Ablation study results after running the above code:" in prompt
        assert "variant A: 0.85\nvariant B: 0.78" in prompt
        assert "# Your task" in prompt
        assert "Summarize the result of ablation study based on the code" in prompt

    def test_summarizer_invokes_debugger_when_ablation_fails(
        self, runner: SubprocessRunner
    ) -> None:
        debugger = DebuggerAgent(runner=runner, model="test")
        repaired_code = (
            "print('Final Validation Performance: 0.80')\n"
            "print('variant_fixed: 0.85')\n"
        )

        def mock_debugger_response(messages, info):
            return ModelResponse(parts=[TextPart(content=f"```python\n{repaired_code}\n```")])

        with debugger.agent.override(model=FunctionModel(function=mock_debugger_response)):
            summarizer = AblationSummarizerAgent(runner=runner, debugger=debugger, model="test")
            with summarizer.agent.override(model=TestModel(custom_output_args=self._report_args())):
                broken_code = "raise RuntimeError('Ablation broken')"
                report = summarizer.summarize(broken_code, run_id="test_debug_run")

        assert isinstance(report, AblationReport)

    def test_summarizer_debugger_exhausted_falls_back_gracefully(
        self, runner: SubprocessRunner
    ) -> None:
        debugger = DebuggerAgent(runner=runner, model="test", max_debug_rounds=1)

        def mock_debugger_unsuccessful(messages, info):
            return ModelResponse(parts=[TextPart(content="```python\nraise RuntimeError('Still broken')\n```")])

        with debugger.agent.override(model=FunctionModel(function=mock_debugger_unsuccessful)):
            summarizer = AblationSummarizerAgent(runner=runner, debugger=debugger, model="test")
            broken_code = "raise RuntimeError('Initial crash')"
            report = summarizer.summarize(broken_code, run_id="test_debug_exhaust")

        assert isinstance(report, AblationReport)
        assert "RuntimeError" in report.raw_log_summary or report.baseline_score == 0.0

    def test_heuristic_report_selects_largest_degradation(
        self, summarizer: AblationSummarizerAgent
    ) -> None:
        raw_output = (
            "Final Validation Performance: 0.80\n"
            "feature_engineering: 0.79\n"
            "model_architecture: 0.50\n"
            "loss_function: 0.75\n"
        )
        report = summarizer._heuristic_report(ablation_code="pass", raw_output=raw_output)
        assert report.baseline_score == pytest.approx(0.80)
        assert len(report.ablation_results) == 3
        # model_architecture caused the biggest drop (-0.30), so it should be the highest impact
        assert report.highest_impact_component == "model_architecture"

    def test_summarizer_places_solution_py_in_ablation_sandbox(
        self, summarizer: AblationSummarizerAgent
    ) -> None:
        merged_solution = (
            "SOLUTION_FLAG = 'merged_v0'\n"
            "print('from solution.py: Final Validation Performance: 0.80')\n"
        )
        ablation_script = (
            "import solution\n"
            "print(f'Imported solution flag: {solution.SOLUTION_FLAG}')\n"
            "print('Final Validation Performance: 0.80')\n"
            "print('variant_A: 0.72')\n"
        )
        def exploding_model(messages, info):
            raise RuntimeError("LLM fallback to heuristic")

        with summarizer.agent.override(model=FunctionModel(function=exploding_model)):
            report = summarizer.summarize(
                ablation_code=ablation_script,
                solution_code=merged_solution,
                run_id="test_solution_import",
                iteration_index=0,
            )
        assert isinstance(report, AblationReport)
        assert "Imported solution flag: merged_v0" in report.raw_log_summary
        sandbox = Path(summarizer.runner.runs_dir) / "test_solution_import" / "sandbox_ablation_t0"
        assert (sandbox / "solution.py").is_file()
        assert (sandbox / "solution.py").read_text(encoding="utf-8") == merged_solution
        assert (sandbox / "ablation.py").is_file()
