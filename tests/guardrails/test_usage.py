"""Unit tests for the data usage guardrail checker."""

from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.guardrails import DataUsageStatus
from problem_2_v2.contracts.task import TaskSpecification
from problem_2_v2.guardrails.usage import DataUsageCheckerAgent

CODE = (
    "import pandas as pd\n"
    "train = pd.read_csv('./input/train.csv')\n"
    "X = train.drop(columns=['label'])\n"
    "y = train['label']\n"
    "model = LogisticRegression()\n"
    "model.fit(X, y)\n"
    "print('Final Validation Performance: 0.80')\n"
)

IMPROVED_CODE = (
    "import pandas as pd\n"
    "train = pd.read_csv('./input/train.csv')\n"
    "user = pd.read_csv('./input/user_features.csv')\n"
    "train = train.merge(user, on='user_id')\n"
    "X = train.drop(columns=['label'])\n"
    "y = train['label']\n"
    "model = LogisticRegression()\n"
    "model.fit(X, y)\n"
    "print('Final Validation Performance: 0.82')\n"
)


def _spec(dataset_files: list[str] | None = None) -> TaskSpecification:
    return TaskSpecification.from_markdown(
        "**Task Name:** Demo\n"
        "**Task Type:** TABULAR_CLASSIFICATION\n"
        "**Metric Name:** AUROC\n"
        "**Metric Direction:** MAXIMIZE\n"
        "**Target Variable:** label\n"
        "**Dataset Files:** " + ", ".join(dataset_files or ["train.csv"]) + "\n"
        "**Description:** classify.\n",
        dataset_dir="/data",
    )


ALL_USED_TEXT = "All the provided information is used."


class TestDataUsageCheckerAgent:
    """Test Figure 22 usage auditing with heuristic fallback."""

    def test_reports_all_data_used(self) -> None:
        checker = DataUsageCheckerAgent(model="test")
        with checker.agent.override(model=TestModel(custom_output_text=ALL_USED_TEXT)):
            status = checker.audit(_spec(["train.csv"]), CODE)
        assert isinstance(status, DataUsageStatus)
        assert status.all_data_used is True
        assert status.missing_sources == []
        assert status.improved_code_block is None

    def test_flags_missing_files_despite_llm_verdict(self) -> None:
        checker = DataUsageCheckerAgent(model="test")
        spec = _spec(["train.csv", "user_features.csv"])
        with checker.agent.override(model=TestModel(custom_output_text=ALL_USED_TEXT)):
            status = checker.audit(spec, CODE)
        assert status.all_data_used is False
        assert status.missing_sources == ["user_features.csv"]

    def test_accepts_improved_code_from_llm(self) -> None:
        checker = DataUsageCheckerAgent(model="test")
        spec = _spec(["train.csv", "user_features.csv"])
        with checker.agent.override(
            model=TestModel(custom_output_text=f"```python\n{IMPROVED_CODE}\n```")
        ):
            status = checker.audit(spec, CODE)
        assert status.all_data_used is False
        assert status.improved_code_block is not None
        assert "```" not in status.improved_code_block
        assert "user_features.csv" in status.improved_code_block
        assert status.missing_sources == []

    def test_prompt_contains_task_and_code(self) -> None:
        checker = DataUsageCheckerAgent(model="test")
        captured: dict[str, str] = {}

        def capturing_model(messages, info):
            captured["prompt"] = messages[-1].parts[0].content
            return ModelResponse(parts=[TextPart(content=ALL_USED_TEXT)])

        spec = _spec(["train.csv"])
        with checker.agent.override(model=FunctionModel(function=capturing_model)):
            checker.audit(spec, CODE)
        assert "train.csv" in captured["prompt"]
        assert "LogisticRegression" in captured["prompt"]
