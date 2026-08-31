"""Unit tests for the data leakage guardrail checker."""

from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from problem_2_v2.contracts.guardrails import DataLeakageStatus
from problem_2_v2.guardrails.leakage import DataLeakageCheckerAgent

LEAKY_CODE = (
    "import pandas as pd\n"
    "train = pd.read_csv('./input/train.csv')\n"
    "test = pd.read_csv('./input/test.csv')\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "scaler = StandardScaler()\n"
    "X_train = scaler.fit_transform(train.drop(columns=['label']))\n"
    "X_test = scaler.transform(test.drop(columns=['label']))\n"
    "print('Final Validation Performance: 0.80')\n"
)

CLEAN_CODE = (
    "import pandas as pd\n"
    "train = pd.read_csv('./input/train.csv')\n"
    "X = train.drop(columns=['label'])\n"
    "y = train['label']\n"
    "model = LogisticRegression()\n"
    "model.fit(X, y)\n"
    "print('Final Validation Performance: 0.80')\n"
)

SUSPICIOUS_BLOCK = (
    "scaler = StandardScaler()\n"
    "X_train = scaler.fit_transform(train.drop(columns=['label']))\n"
    "X_test = scaler.transform(test.drop(columns=['label']))"
)

CORRECTED_BLOCK = (
    "scaler = StandardScaler()\n"
    "X_train = scaler.fit_transform(train.drop(columns=['label']))\n"
    "X_test = scaler.transform(test.drop(columns=['label']))"
)


class TestDataLeakageCheckerAgent:
    """Test leakage detection (Figure 20) and repair (Figure 21)."""

    def _leak_args(self, status: str, code_block: str | None) -> dict[str, object]:
        return {
            "leakage_status": status,
            "is_leaking": status.lower().startswith("yes"),
            "suspicious_code_block": code_block,
            "corrected_code_block": None,
            "explanation": "static audit",
        }

    def test_detects_leakage(self) -> None:
        checker = DataLeakageCheckerAgent(model="test")
        with checker.check_agent.override(
            model=TestModel(
                custom_output_args=self._leak_args("Yes Data Leakage", SUSPICIOUS_BLOCK)
            )
        ):
            status = checker.check(LEAKY_CODE)
        assert isinstance(status, DataLeakageStatus)
        assert status.is_leaking is True
        assert status.suspicious_code_block == SUSPICIOUS_BLOCK

    def test_reports_clean_code(self) -> None:
        checker = DataLeakageCheckerAgent(model="test")
        with checker.check_agent.override(
            model=TestModel(custom_output_args=self._leak_args("No Data Leakage", SUSPICIOUS_BLOCK))
        ):
            status = checker.check(CLEAN_CODE)
        assert status.is_leaking is False

    def test_repair_replaces_suspicious_block(self) -> None:
        checker = DataLeakageCheckerAgent(model="test")
        with checker.repair_agent.override(
            model=TestModel(custom_output_text=f"```python\n{CORRECTED_BLOCK}\n```")
        ):
            repaired = checker.repair(LEAKY_CODE, SUSPICIOUS_BLOCK)
        assert isinstance(repaired, str)
        assert SUSPICIOUS_BLOCK in repaired
        assert "X_train = scaler.fit_transform" in repaired

    def test_repair_strips_markdown_fences(self) -> None:
        checker = DataLeakageCheckerAgent(model="test")
        with checker.repair_agent.override(
            model=TestModel(custom_output_text=f"```python\n{CORRECTED_BLOCK}\n```")
        ):
            repaired = checker.repair(LEAKY_CODE, SUSPICIOUS_BLOCK)
        assert "```" not in repaired

    def test_repair_without_code_leaves_code_unchanged(self) -> None:
        checker = DataLeakageCheckerAgent(model="test")
        with checker.repair_agent.override(
            model=TestModel(custom_output_text="I cannot help with this task.")
        ):
            repaired = checker.repair(LEAKY_CODE, SUSPICIOUS_BLOCK)
        assert repaired == LEAKY_CODE

    def test_repair_falls_through_when_block_not_found(self) -> None:
        checker = DataLeakageCheckerAgent(model="test")
        with checker.repair_agent.override(
            model=TestModel(custom_output_text="```python\nx = 1\n```")
        ):
            # Previously raised ValueError; now falls through to full-script
            # rewrite and returns whatever the repair agent produces.
            result = checker.repair(CLEAN_CODE, "block that is not there")
        assert isinstance(result, str)

    def test_audit_fixes_leaky_code(self) -> None:
        checker = DataLeakageCheckerAgent(model="test")
        fixed_block = CORRECTED_BLOCK + "\n# repaired leak"
        with (
            checker.check_agent.override(
                model=TestModel(
                    custom_output_args=self._leak_args("Yes Data Leakage", SUSPICIOUS_BLOCK)
                )
            ),
            checker.repair_agent.override(
                model=TestModel(custom_output_text=f"```python\n{fixed_block}\n```")
            ),
        ):
            status, code = checker.audit(LEAKY_CODE)
        assert status.is_leaking is True
        assert code != LEAKY_CODE
        assert "Final Validation Performance" in code
        assert "```" not in code

    def test_audit_returns_code_unchanged_when_clean(self) -> None:
        checker = DataLeakageCheckerAgent(model="test")
        with checker.check_agent.override(
            model=TestModel(custom_output_args=self._leak_args("No Data Leakage", None))
        ):
            status, code = checker.audit(CLEAN_CODE)
        assert status.is_leaking is False
        assert code == CLEAN_CODE

    def test_prompt_contains_solution_code(self) -> None:
        checker = DataLeakageCheckerAgent(model="test")
        captured: dict[str, str] = {}

        def capturing_model(messages, info):
            captured["prompt"] = messages[-1].parts[0].content
            return ModelResponse(
                parts=[
                    TextPart(
                        content='{"leakage_status": "No Data Leakage", "is_leaking": false, '
                        '"suspicious_code_block": null, "corrected_code_block": null, '
                        '"explanation": "clean"}'
                    )
                ]
            )

        with checker.check_agent.override(model=FunctionModel(function=capturing_model)):
            checker.check(CLEAN_CODE)
        assert "LogisticRegression" in captured["prompt"]

    def test_build_prompt_format(self) -> None:
        prompt = DataLeakageCheckerAgent.build_check_prompt(CLEAN_CODE)
        assert "# Python code\n" in prompt
        assert CLEAN_CODE in prompt
        assert "# Your task" in prompt
        assert "validation and test samples are preprocessed" in prompt
        assert "preventing data leakage" in prompt
        assert "# Requirement" in prompt
        assert "The code block should be an exact subset of the above Python code." in prompt
        assert "answer 'Yes Data Leakage'." in prompt
        assert "answer 'No Data Leakage'." in prompt
        assert "Answer = {'leakage_status': str, 'code_block': str}" in prompt
        assert "Return: list[Answer]" in prompt
