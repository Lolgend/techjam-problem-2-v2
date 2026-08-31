"""Candidate evaluation agent ($A_init$): code generation and ranking.

Generates a Python script for each retrieved model card
(Figure 10 prompt), enforces single-file/hold-out/30k-subsample
constraints, executes it in the sandbox with autonomous debugging, and
ranks the candidates into the descending performance permutation used by
the merging stage.
"""

from __future__ import annotations

import logfire
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent

from problem_2_v2.contracts.code_utils import extract_python_code, validate_python_syntax
from problem_2_v2.contracts.enums import MetricDirection
from problem_2_v2.contracts.search import ModelCard
from problem_2_v2.contracts.task import ExecutionResult, TaskSpecification
from problem_2_v2.runner.debugger import DebuggerAgent

_EVALUATOR_PROMPT_TEMPLATE = (
    "# Introduction\n"
    "- You are a Kaggle grandmaster attending a competition.\n"
    "- We will now provide a task description and a model description.\n"
    "- You need to implement your Python solution using the provided model.\n"
    "# Task description\n"
    "{task_description}\n"
    "# Model description\n"
    "## Model name\n"
    "{model_description}\n"
    "## Example Python code\n"
    "{example_code}\n"
    "# Your task\n"
    "- Implement the solution in Python.\n"
    "- You must use the model as described in the model description.\n"
    "- If it is the official baseline, ignore everything and just copy the official baseline code as is. "
    "Retain the original form, it is tested and runs.\n"
    "- This first solution design should be relatively simple, without ensembling or "
    "hyper-parameter optimization.\n"
    "- All the provided data is already prepared and available in the `./input` directory. "
    "There is no need to unzip any files.\n"
    "- Do not include other models that are not directly related to the model described.\n"
    "- [Does not apply to official baseline] Use PyTorch rather than TensorFlow. Use CUDA if you need. All the necessary "
    "libraries are installed.\n"
    "- Only use the provided train data in the `./input` directory. Do not load test data "
    "during validation.\n"
    "- If there are more than 30,000 training samples, you must subsample to 30,000 for a "
    "faster run.\n"
    "# [IMPORTANT] Mandatory Evaluation Protocol\n"
    "- You MUST ONLY evaluate validation performance using the official evaluation harness "
    "and nothing else: 'from evaluate import evaluate'. This import will not fail\n"
    "- The evaluation function signature is: evaluate(user_ids, labels, scores, k=5)\n"
    "  where:\n"
    "    * user_ids: 1D list/array of validation user IDs (e.g. valid_df['user_id'].tolist())\n"
    "    * labels: 1D list/array of validation binary labels 0/1 (e.g. "
    "valid_df['long_view'].tolist())\n"
    "    * scores: 1D list/array of continuous real-valued prediction scores/logits/probabilities "
    "from the model\n"
    "  It returns a dict with keys 'GAUC', 'nDCG@5', and 'primary' (the mean of GAUC and "
    "nDCG@5).\n"
    "- Extract val_res['primary'] and print: print(f'Final Validation Performance: "
    "{{val_res[\"primary\"]:.6f}}')\n"
    "- Do NOT write custom metric calculations (like standard roc_auc_score, accuracy, or "
    "loss); you MUST call evaluate().\n"
    "# Required\n"
    "- There should be no additional headings or text in your response.\n"
    "- Print out or return a final performance metric in your answer in a clear format with "
    "the exact words: 'Final Validation Performance: {{final_validation_score}}'.\n"
    "- The code should be a single-file Python program that can be executed as-is.\n"
    "- Your response should only contain a single code block.\n"
    "- Do not use exit() function in the Python code.\n"
    "- Do not use try: and except: or if else to ignore unintended behavior.\n"
)


class CandidateEvaluation(BaseModel):

    """Outcome of evaluating a single candidate model.

    Attributes:
        model_name: Name of the evaluated candidate model.
        code: Final (possibly debugged) executable script.
        result: Final execution result, if execution was attempted.
        debug_rounds: Number of debugger repair rounds used.
        score: Parsed validation score, or ``None`` on failure.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model_name: str = Field(description="Evaluated candidate model name.")
    code: str = Field(default="", description="Final executable script.")
    result: ExecutionResult | None = Field(default=None, description="Final execution result.")
    debug_rounds: int = Field(default=0, description="Debugger repair rounds used.")
    score: float | None = Field(default=None, description="Validation score, or None on failure.")


class CandidateEvaluatorAgent:
    """Generates, executes, and ranks candidate solutions for retrieved models.

    Attributes:
        debugger: Autonomous debugger used to repair failing scripts.
        agent: Pydantic AI agent producing candidate Python source.
    """

    def __init__(
        self,
        debugger: DebuggerAgent,
        model: str = "openai:gpt-4o",
    ) -> None:
        """Create a candidate evaluator.

        Args:
            debugger: Debugger agent for execution repair fallback.
            model: Pydantic AI model string.
        """
        self.debugger = debugger
        self.agent = Agent(
            model,
            name="candidate_evaluator_agent",
            output_type=str,
            defer_model_check=True,
        )

    def evaluate(
        self,
        spec: TaskSpecification,
        card: ModelCard,
        run_id: str,
        candidate_id: str,
    ) -> CandidateEvaluation:
        """Generate and evaluate one candidate model script.

        Args:
            spec: The task specification.
            card: The candidate model card.
            run_id: Identifier of the current run.
            candidate_id: Identifier of this candidate.

        Returns:
            A ``CandidateEvaluation`` with the executable code, execution
            result, and parsed validation score.
        """
        prompt = self._build_prompt(spec, card)
        with logfire.span("evaluator.generate", model=card.model_name):
            response = self.agent.run_sync(prompt)
        code = extract_python_code(response.output)

        valid, error = validate_python_syntax(code)
        if not valid or not code:
            logfire.warn("evaluator.invalid_code", model=card.model_name, error=error)
            return CandidateEvaluation(
                model_name=card.model_name,
                code=code,
                result=ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=f"Invalid generated code: {error}",
                    returncode=-1,
                    duration_seconds=0.0,
                ),
                debug_rounds=0,
                score=None,
            )

        outcome = self.debugger.debug(
            code,
            run_id=run_id,
            candidate_id=candidate_id,
            dataset_dir=spec.dataset_dir,
            dataset_files=spec.dataset_files,
        )
        return CandidateEvaluation(
            model_name=card.model_name,
            code=outcome.code,
            result=outcome.result,
            debug_rounds=outcome.debug_rounds,
            score=outcome.result.validation_score,
        )

    def evaluate_all(
        self,
        spec: TaskSpecification,
        cards: list[ModelCard],
        run_id: str,
    ) -> list[CandidateEvaluation]:
        """Evaluate every candidate model card.

        Args:
            spec: The task specification.
            cards: The candidate model cards to evaluate.
            run_id: Identifier of the current run.

        Returns:
            A list of evaluations in the same order as ``cards``.
        """
        return [
            self.evaluate(spec, card, run_id=run_id, candidate_id=f"cand{i + 1}")
            for i, card in enumerate(cards)
        ]

    @staticmethod
    def ranking(
        evaluations: list[CandidateEvaluation],
        direction: MetricDirection,
    ) -> list[CandidateEvaluation]:
        """Sort evaluations into the descending performance permutation.

        Scored candidates are ordered best-first under the metric
        direction; evaluations without a score are sorted last.

        Args:
            evaluations: Evaluations to rank.
            direction: Metric direction used for comparison.

        Returns:
            The ranked list (best candidate first).
        """
        scored = [e for e in evaluations if e.score is not None]
        failed = [e for e in evaluations if e.score is None]

        def score_key(evaluation: CandidateEvaluation) -> float:
            return evaluation.score if evaluation.score is not None else float("-inf")

        scored.sort(key=score_key, reverse=direction is MetricDirection.MAXIMIZE)
        return scored + failed

    @staticmethod
    def build_prompt(spec: TaskSpecification, card: ModelCard) -> str:
        """Build the candidate generation prompt."""
        if spec.raw_description:
            task_description = spec.raw_description
        else:
            task_desc_parts: list[str] = []
            if spec.task_name:
                task_desc_parts.append(spec.task_name)
            if spec.task_type:
                task_desc_parts.append(spec.task_type.value)
            if spec.description:
                task_desc_parts.append(spec.description)
            if spec.metric_name:
                task_desc_parts.append(f"Evaluation metric: {spec.metric_name}")
            if spec.target_variable:
                task_desc_parts.append(f"Target variable: {spec.target_variable}")
            if spec.dataset_files:
                task_desc_parts.append(f"Dataset files: {', '.join(spec.dataset_files)}")
            if spec.constraints:
                task_desc_parts.append(f"Constraints: {spec.constraints}")
            task_description = "\n".join(task_desc_parts).strip()

        model_description = card.model_name
        if card.rationale:
            model_description = f"{card.model_name}\n{card.rationale}"

        return _EVALUATOR_PROMPT_TEMPLATE.format(
            task_description=task_description,
            model_description=model_description,
            example_code=card.example_code,
        )

    _build_prompt = build_prompt

