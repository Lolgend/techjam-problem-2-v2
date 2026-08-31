"""Ablation study generation and summarization agents.

``AblationAgent`` ($A_abl$) generates ablation scripts that modify or
disable 2-3 pipeline components (Figure 12 prompt). ``AblationSummarizerAgent``
($A_summarize$) executes the ablation script in the sandbox and digests the
raw output into a structured ``AblationReport`` (Figure 13 prompt), with a
deterministic heuristic fallback when the LLM is unavailable.
"""

from __future__ import annotations

import re
from typing import ClassVar

import logfire
from pydantic_ai import Agent

from problem_2_v2.contracts.code_utils import extract_python_code
from problem_2_v2.contracts.refinement import (
    AblationReport,
    AblationResultItem,
)
from problem_2_v2.runner.sandbox import SubprocessRunner

_ABLATION_PROMPT_TEMPLATE = (
    "# Introduction\n"
    "- You are a Kaggle grandmaster attending a competition.\n"
    "- In order to win this competition, you need to perform an ablation study on the current\n"
    "Python solution to know which parts of the code contribute the most to the overall\n"
    "performance.\n"
    "- We will now provide a current Python solution.\n"
    "- We will also provide the summaries of previous ablation studies.\n"
    "# Python solution\n"
    "{solution_script}\n"
    "{previous_ablations}"
    "# Instructions\n"
    "- You need you to generate a simple Python code that performs an ablation study on the\n"
    "train.py script.\n"
    "- The generated code should create variations by modifying or disabling parts (2-3 parts)\n"
    "of the training process.\n"
    "- Your ablation study should concentrate on the other parts that have not been previously\n"
    "considered.\n"
    "- For each ablation, print out how the modification affects the model's performance.\n"
    "# Response format\n"
    "- There should be no additional headings or text in your response.\n"
    "- The Python code for the ablation study should not load test data. It should only focus on\n"
    "training and evaluating the model on the validation set.\n"
    "- The code should include a printing statement that shows the performance of each ablation.\n"
    "- The code should consequently print out what part of the code contributes the most to the\n"
    "overall performance."
)

_SUMMARIZE_PROMPT_TEMPLATE = (
    "# Your code for ablation study was:\n"
    "{ablation_code}\n"
    "# Ablation study results after running the above code:\n"
    "{raw_output}\n"
    "# Your task\n"
    "- Summarize the result of ablation study based on the code and printed output."
)

_SUMMARIZE_INSTRUCTIONS = (
    "You summarize the result of an ablation study based on the ablation "
    "code and its printed output.\n"
    "- Identify each ablation variant and its reported performance.\n"
    "- Determine the baseline score and the delta of each variant.\n"
    "- Identify the single highest-impact pipeline component.\n"
    "- Keep the raw log summary concise and faithful to the output."
)


class AblationAgent:
    """Generates ablation study scripts for the current solution.

    Attributes:
        agent: Pydantic AI agent producing the ablation script.
    """

    def __init__(self, model: str = "openai:gpt-4o") -> None:
        """Create an ablation generation agent.

        Args:
            model: Pydantic AI model string.
        """
        self.agent = Agent(
            model,
            name="ablation_agent",
            output_type=str,
            defer_model_check=True,
        )

    @staticmethod
    def build_prompt(solution: str, previous_ablations: list[str]) -> str:
        """Build the Figure 12 ablation study prompt.

        Args:
            solution: The current solution script.
            previous_ablations: Summaries of previous ablation studies.

        Returns:
            The formatted ablation prompt string.
        """
        history_parts = [
            f"## Previous ablation study result {{{i}}}\n{summary}"
            for i, summary in enumerate(previous_ablations)
        ]
        history_joined = "\n".join(history_parts)
        previous_str = f"{history_joined}\n" if history_parts else ""
        return _ABLATION_PROMPT_TEMPLATE.format(
            solution_script=solution,
            previous_ablations=previous_str,
        )

    _build_prompt = build_prompt

    def generate_ablation(
        self,
        solution: str,
        previous_ablations: list[str],
    ) -> str:
        """Generate an executable ablation study script.

        Args:
            solution: The current solution script.
            previous_ablations: Summaries of previous ablation studies.

        Returns:
            The cleaned ablation script source (fences stripped).
        """
        prompt = self.build_prompt(solution=solution, previous_ablations=previous_ablations)
        with logfire.span("ablation.generate"):
            response = self.agent.run_sync(prompt)
        return extract_python_code(response.output)


class AblationSummarizerAgent:
    """Executes ablation scripts and summarizes results into reports.

    Attributes:
        runner: Sandbox runner used to execute ablation scripts.
        agent: Pydantic AI agent producing the structured ``AblationReport``.
    """

    _SCORE_LINE_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"^.*?(?P<name>[A-Za-z0-9_ \t-]+?)\s*[:=]\s*(?P<score>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
    )

    def __init__(self, runner: SubprocessRunner, model: str = "openai:gpt-4o") -> None:
        """Create an ablation summarizer.

        Args:
            runner: Sandbox runner for ablation script execution.
            model: Pydantic AI model string.
        """
        self.runner = runner
        self.agent = Agent(
            model,
            name="ablation_summarizer_agent",
            output_type=AblationReport,
            instructions=_SUMMARIZE_INSTRUCTIONS,
            defer_model_check=True,
        )

    @staticmethod
    def build_prompt(ablation_code: str, raw_output: str) -> str:
        """Build the Figure 13 ablation summarization prompt.

        Args:
            ablation_code: The ablation study script that was executed.
            raw_output: Raw stdout/stderr execution output.

        Returns:
            The formatted summarization prompt string.
        """
        return _SUMMARIZE_PROMPT_TEMPLATE.format(
            ablation_code=ablation_code,
            raw_output=raw_output or "(no output)",
        )

    _build_prompt = build_prompt

    def summarize(
        self,
        ablation_code: str,
        run_id: str,
        dataset_dir: str | None = None,
        dataset_files: list[str] | None = None,
        iteration_index: int | None = None,
    ) -> AblationReport:
        """Execute the ablation script and summarize its raw output.

        Args:
            ablation_code: The ablation study script to execute.
            run_id: Identifier of the current run.
            dataset_dir: Dataset directory for the sandbox, if any.
            dataset_files: Dataset files to map, if any.
            iteration_index: Outer-loop iteration index, used to scope the
                ablation sandbox to ``sandbox_ablation_t{t}`` so repeated
                outer iterations never collide on one sandbox.

        Returns:
            A structured ``AblationReport`` describing per-variant outcomes
            and the highest-impact component.
        """
        candidate_id = f"ablation_t{iteration_index}" if iteration_index is not None else "ablation"
        sandbox = self.runner.prepare_sandbox(
            run_id=run_id,
            candidate_id=candidate_id,
            dataset_dir=dataset_dir,
            dataset_files=dataset_files,
        )
        with logfire.span("ablation.execute"):
            result = self.runner.run_code(ablation_code, sandbox_dir=str(sandbox))
        raw_output = result.stdout or result.stderr

        try:
            with logfire.span("ablation.summarize_llm"):
                prompt = self.build_prompt(ablation_code=ablation_code, raw_output=raw_output)
                response = self.agent.run_sync(prompt)
            return response.output
        except Exception:
            logfire.warn("ablation.summarize_llm.failed; using heuristic parser")
            return self._heuristic_report(ablation_code, raw_output)

    def _heuristic_report(self, ablation_code: str, raw_output: str) -> AblationReport:
        """Build a report from raw output without LLM involvement."""
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        baseline = 0.0
        results: list[AblationResultItem] = []

        for line in lines:
            match = self._SCORE_LINE_RE.match(line)
            if match is None:
                continue
            name = match.group("name").strip()
            try:
                score = float(match.group("score"))
            except ValueError:
                continue
            if "final validation performance" in name.lower():
                baseline = score
                continue
            results.append(
                AblationResultItem(
                    variant_id=name,
                    validation_score=score,
                    delta_from_baseline=score - baseline,
                    summary=line,
                )
            )

        top = max(results, key=lambda item: item.delta_from_baseline) if results else None
        return AblationReport(
            baseline_score=baseline,
            ablation_results=results,
            highest_impact_component=top.variant_id if top else "",
            raw_log_summary=raw_output[-2000:],
        )
