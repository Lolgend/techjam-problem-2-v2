"""Unit tests for the targeted code block extractor agent."""

import pytest
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.enums import ComponentCategory
from problem_2_v2.contracts.refinement import RefinementPlan, TargetCodeBlock
from problem_2_v2.refinement.extractor import CodeBlockExtractorAgent

SOLUTION = (
    "import pandas as pd\n"
    "train = pd.read_csv('./input/train.csv')\n"
    "X = train.drop(columns=['label'])\n"
    "y = train['label']\n"
    "model = LogisticRegression()\n"
    "model.fit(X, y)\n"
    "print('Final Validation Performance: 0.80')\n"
)

SOLUTION_WITH_DEF = (
    "import pandas as pd\n"
    "def train_model():\n"
    "    X = data.drop(columns=['y'])\n"
    "    y = data['y']\n"
    "    model = LogisticRegression()\n"
    "    model.fit(X, y)\n"
    "    return model\n"
    "print('Final Validation Performance: 0.80')\n"
)

TARGET_BLOCK = "model = LogisticRegression()\nmodel.fit(X, y)"

PLAN_TEXT = (
    "Replace the logistic regression with gradient boosted trees tuned on the validation set."
)


class TestCodeBlockExtractorAgent:
    """Test Figure 14 target extraction and initial plan generation."""

    def _item_args(
        self, code_block: str, category: str = "MODEL_ARCHITECTURE"
    ) -> dict[str, object]:
        return {
            "code_block": code_block,
            "plan": PLAN_TEXT,
            "category": category,
        }

    def test_extracts_target_block_and_plan(self) -> None:
        agent = CodeBlockExtractorAgent(model="test")
        with agent.agent.override(
            model=TestModel(custom_output_args=[self._item_args(TARGET_BLOCK)])
        ):
            block, plan = agent.extract(
                solution=SOLUTION,
                ablation_summary="Model architecture had the biggest impact.",
                previous_blocks=[],
            )
        assert isinstance(block, TargetCodeBlock)
        assert block.raw_code == TARGET_BLOCK
        assert block.category is ComponentCategory.MODEL_ARCHITECTURE
        assert isinstance(plan, RefinementPlan)
        assert plan.natural_language_plan == PLAN_TEXT
        assert plan.iteration_index == 0

    def test_extracts_indented_block_from_script(self) -> None:
        agent = CodeBlockExtractorAgent(model="test")
        solution = (
            "def train():\n    X = data.drop(columns=['y'])\n    y = data['y']\n    return X, y\n"
        )
        block_text = "X = data.drop(columns=['y'])\n    y = data['y']"
        with agent.agent.override(
            model=TestModel(custom_output_args=[self._item_args(block_text)])
        ):
            block, _ = agent.extract(
                solution=solution,
                ablation_summary="Features mattered most.",
                previous_blocks=[],
            )
        assert block.raw_code == "    X = data.drop(columns=['y'])\n    y = data['y']"

    def test_raises_when_agent_returns_no_items(self) -> None:
        agent = CodeBlockExtractorAgent(model="test")
        with (
            agent.agent.override(model=TestModel(custom_output_args=[])),
            pytest.raises(ValueError, match="no refinement plans"),
        ):
            agent.extract(
                solution=SOLUTION,
                ablation_summary="summary",
                previous_blocks=[],
            )

    def test_accepts_llm_quote_variation(self) -> None:
        agent = CodeBlockExtractorAgent(model="test")
        block_text = 'X = train.drop(columns=["label"])\ny = train["label"]'
        with agent.agent.override(
            model=TestModel(custom_output_args=[self._item_args(block_text)])
        ):
            block, _ = agent.extract(
                solution=SOLUTION,
                ablation_summary="Features mattered most.",
                previous_blocks=[],
            )
        assert block.raw_code == "X = train.drop(columns=['label'])\ny = train['label']"

    def test_accepts_llm_trailing_comment_variation(self) -> None:
        agent = CodeBlockExtractorAgent(model="test")
        block_text = "model = LogisticRegression()\nmodel.fit(X, y)  # fit on all rows"
        with agent.agent.override(
            model=TestModel(custom_output_args=[self._item_args(block_text)])
        ):
            block, _ = agent.extract(
                solution=SOLUTION,
                ablation_summary="Model mattered most.",
                previous_blocks=[],
            )
        assert block.raw_code == "model = LogisticRegression()\nmodel.fit(X, y)"

    def test_ast_fallback_returns_primary_def_block(self) -> None:
        agent = CodeBlockExtractorAgent(model="test")
        block_text = (
            "def train_model():\n"
            '    model = LogisticRegression(penalty="l2")\n'
            "    model.fit(X, y)\n"
            "    return model"
        )
        with agent.agent.override(
            model=TestModel(custom_output_args=[self._item_args(block_text)])
        ):
            block, _ = agent.extract(
                solution=SOLUTION_WITH_DEF,
                ablation_summary="Model architecture mattered most.",
                previous_blocks=[],
            )
        assert block.raw_code == (
            "def train_model():\n"
            "    X = data.drop(columns=['y'])\n"
            "    y = data['y']\n"
            "    model = LogisticRegression()\n"
            "    model.fit(X, y)\n"
            "    return model"
        )

    def test_falls_back_to_solution_when_no_defs_and_no_match(self) -> None:
        agent = CodeBlockExtractorAgent(model="test")
        block_text = "model = RandomForest()\nmodel.fit(X, y)"
        with agent.agent.override(
            model=TestModel(custom_output_args=[self._item_args(block_text)])
        ):
            block, _ = agent.extract(
                solution=SOLUTION,
                ablation_summary="Model mattered most.",
                previous_blocks=[],
            )
        assert block.raw_code == SOLUTION

    def test_prompt_includes_history_and_previous_blocks(self) -> None:
        agent = CodeBlockExtractorAgent(model="test")
        prompt = agent.build_prompt(
            solution=SOLUTION,
            ablation_summary="Imputation mattered most.",
            previous_blocks=["imputer = SimpleImputer(strategy='mean')"],
        )
        assert "Imputation mattered most." in prompt
        assert "imputer = SimpleImputer(strategy='mean')" in prompt
        assert "Python solution" in prompt
        assert "2-3" not in prompt
