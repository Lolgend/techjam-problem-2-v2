"""Unit tests for the coder agent and script patching helper."""

import pytest
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.enums import ComponentCategory
from problem_2_v2.contracts.refinement import RefinementPlan, TargetCodeBlock
from problem_2_v2.refinement.coder import CoderAgent, patch_script

TARGET_BLOCK = "model = LogisticRegression()\nmodel.fit(X, y)"
REFINED_BLOCK = (
    "from xgboost import XGBClassifier\n"
    "model = XGBClassifier(n_estimators=200, max_depth=6)\n"
    "model.fit(X, y)"
)
PLAN_TEXT = "Replace logistic regression with gradient boosted trees."

SCRIPT = (
    "import pandas as pd\n"
    "train = pd.read_csv('./input/train.csv')\n"
    "X = train.drop(columns=['label'])\n"
    "y = train['label']\n"
    "model = LogisticRegression()\n"
    "model.fit(X, y)\n"
    "print('Final Validation Performance: 0.80')\n"
)

INDENTED_SCRIPT = (
    "def train_model():\n"
    "    X = data.drop(columns=['label'])\n"
    "    y = data['label']\n"
    "    model = LogisticRegression()\n"
    "    model.fit(X, y)\n"
    "    return model\n"
)


def _block() -> TargetCodeBlock:
    return TargetCodeBlock(
        raw_code=TARGET_BLOCK,
        category=ComponentCategory.MODEL_ARCHITECTURE,
        start_line=None,
        end_line=None,
        initial_plan=PLAN_TEXT,
    )


def _plan(k: int = 0) -> RefinementPlan:
    return RefinementPlan(
        plan_id=f"p{k}",
        natural_language_plan=PLAN_TEXT,
        target_subcomponents=["model"],
        expected_gain="+0.02",
        iteration_index=k,
    )


class TestCoderAgent:
    """Test Figure 15 code block transformation."""

    def test_refines_block_and_strips_fences(self) -> None:
        agent = CoderAgent(model="test")
        with agent.agent.override(
            model=TestModel(custom_output_text=f"```python\n{REFINED_BLOCK}\n```")
        ):
            refined = agent.refine(_block(), _plan())
        assert refined == REFINED_BLOCK
        assert "```" not in refined

    def test_prompt_contains_block_and_plan(self) -> None:
        agent = CoderAgent(model="test")
        captured: dict[str, str] = {}

        def capturing_model(messages, info):
            captured["prompt"] = messages[-1].parts[0].content
            return ModelResponse(parts=[TextPart(content=REFINED_BLOCK)])

        with agent.agent.override(model=FunctionModel(function=capturing_model)):
            agent.refine(_block(), _plan(k=2))
        assert TARGET_BLOCK in captured["prompt"]
        assert PLAN_TEXT in captured["prompt"]
        assert "subsampling" in captured["prompt"]

    def test_instructions_mandate_official_evaluate_harness(self) -> None:
        from problem_2_v2.refinement.coder import _CODER_INSTRUCTIONS

        assert "from evaluate import evaluate" in _CODER_INSTRUCTIONS


class TestPatchScript:
    """Test AST-safe script patching with whitespace fallback."""

    def test_exact_substring_patch(self) -> None:
        patched = patch_script(SCRIPT, TARGET_BLOCK, REFINED_BLOCK)
        assert TARGET_BLOCK not in patched
        assert "XGBClassifier" in patched
        assert "Final Validation Performance" in patched

    def test_indentation_tolerant_patch(self) -> None:
        patched = patch_script(
            INDENTED_SCRIPT,
            "model = LogisticRegression()\nmodel.fit(X, y)",
            "model = XGBClassifier()\nmodel.fit(X, y)",
        )
        assert "    model = XGBClassifier()" in patched
        assert "    model.fit(X, y)" in patched

    def test_raises_when_block_not_found(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            patch_script(SCRIPT, "block = not_there()", "x = 1")

    def test_raises_when_patch_produces_invalid_syntax(self) -> None:
        with pytest.raises(ValueError, match="invalid Python"):
            patch_script(SCRIPT, TARGET_BLOCK, "if True:")
