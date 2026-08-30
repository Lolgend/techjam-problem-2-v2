"""Unit tests for resilient patching in the data leakage guardrail.

Phase 1 tests: verify that ``_patch()`` succeeds with exact matching,
fuzzy/normalized matching (via ``find_matching_block``), and full-script
rewrite fallback, and that ``repair()`` never raises ``ValueError``.
"""

from pydantic_ai.models.test import TestModel

from problem_2_v2.guardrails.leakage import DataLeakageCheckerAgent

# --- Test fixtures ---

# Script where the suspicious block appears verbatim.
SCRIPT_EXACT = (
    "import pandas as pd\n"
    "train = pd.read_csv('./input/train.csv')\n"
    "test = pd.read_csv('./input/test.csv')\n"
    "scaler = StandardScaler()\n"
    "X_train = scaler.fit_transform(train.drop(columns=['label']))\n"
    "X_test = scaler.transform(test.drop(columns=['label']))\n"
    "print('Final Validation Performance: 0.80')\n"
)

SUSPICIOUS_EXACT = (
    "scaler = StandardScaler()\n"
    "X_train = scaler.fit_transform(train.drop(columns=['label']))\n"
    "X_test = scaler.transform(test.drop(columns=['label']))"
)

CORRECTED_BLOCK = (
    "scaler = StandardScaler()\n"
    "X_train = scaler.fit_transform(train.drop(columns=['label']))\n"
    "X_test = scaler.transform(test.drop(columns=['label']))  # fixed"
)

# Script where the suspicious block has extra indentation (whitespace diff).
SCRIPT_INDENTED = (
    "import pandas as pd\n"
    "def preprocess():\n"
    "    scaler = StandardScaler()\n"
    "    X_train = scaler.fit_transform(train.drop(columns=['label']))\n"
    "    X_test = scaler.transform(test.drop(columns=['label']))\n"
    "    print('done')\n"
)

# The suspicious block extracted *without* the indentation (as LLM would return).
SUSPICIOUS_WHITESPACE = (
    "scaler = StandardScaler()\n"
    "X_train = scaler.fit_transform(train.drop(columns=['label']))\n"
    "X_test = scaler.transform(test.drop(columns=['label']))"
)

# Script where the suspicious block uses double quotes, but the LLM
# extracted it with single quotes.
SCRIPT_DOUBLE_QUOTES = (
    "import pandas as pd\n"
    'train = pd.read_csv("./input/train.csv")\n'
    "scaler = StandardScaler()\n"
    'X_train = scaler.fit_transform(train.drop(columns=["label"]))\n'
    'X_test = scaler.transform(test.drop(columns=["label"]))\n'
    "print('Final Validation Performance: 0.80')\n"
)

SUSPICIOUS_SINGLE_QUOTES = (
    "scaler = StandardScaler()\n"
    "X_train = scaler.fit_transform(train.drop(columns=['label']))\n"
    "X_test = scaler.transform(test.drop(columns=['label']))"
)

# A suspicious block that has no relationship to the script at all
# (forces full-script rewrite fallback).
SCRIPT_UNMATCHED = (
    "import numpy as np\nresult = np.mean([1, 2, 3])\nprint('Final Validation Performance: 0.90')\n"
)

SUSPICIOUS_UNMATCHED = (
    "completely_different_function()\nanother_unrelated_call()\nthird_unknown_line()"
)

FULL_SCRIPT_REWRITE = (
    "import numpy as np\n"
    "result = np.mean([1, 2, 3])\n"
    "# leakage fixed\n"
    "print('Final Validation Performance: 0.90')\n"
)


class TestResilientPatch:
    """Test multi-tier patching in ``DataLeakageCheckerAgent._patch()``."""

    def test_exact_string_match(self) -> None:
        """Tier 1: exact ``str.replace`` succeeds when block is verbatim."""
        checker = DataLeakageCheckerAgent(model="test")
        result = checker._patch(SCRIPT_EXACT, SUSPICIOUS_EXACT, CORRECTED_BLOCK)
        assert "# fixed" in result
        # The original unfixed last line should be replaced by the corrected one.
        assert "X_test = scaler.transform(test.drop(columns=['label']))  # fixed" in result

    def test_fuzzy_whitespace_match(self) -> None:
        """Tier 2: fuzzy match succeeds when block has indent differences."""
        checker = DataLeakageCheckerAgent(model="test")
        result = checker._patch(SCRIPT_INDENTED, SUSPICIOUS_WHITESPACE, CORRECTED_BLOCK)
        # The corrected block should appear (indentation-aligned) in the result
        assert "# fixed" in result
        # Original suspicious content should be replaced
        assert "X_test = scaler.transform(test.drop(columns=['label']))\n    print" not in result

    def test_fuzzy_quote_style_match(self) -> None:
        """Tier 2: fuzzy match succeeds when block has single/double quote diffs."""
        checker = DataLeakageCheckerAgent(model="test")
        result = checker._patch(SCRIPT_DOUBLE_QUOTES, SUSPICIOUS_SINGLE_QUOTES, CORRECTED_BLOCK)
        assert "# fixed" in result

    def test_full_script_rewrite_fallback(self) -> None:
        """Tier 3: when fuzzy match also fails, full-script rewrite is used."""
        checker = DataLeakageCheckerAgent(model="test")
        with checker.repair_agent.override(
            model=TestModel(custom_output_text=f"```python\n{FULL_SCRIPT_REWRITE}\n```")
        ):
            result = checker._patch(SCRIPT_UNMATCHED, SUSPICIOUS_UNMATCHED, "irrelevant_corrected")
        assert "# leakage fixed" in result

    def test_full_script_rewrite_no_code_returns_original(self) -> None:
        """Tier 3: when rewrite produces no extractable code, return original."""
        checker = DataLeakageCheckerAgent(model="test")
        with checker.repair_agent.override(
            model=TestModel(custom_output_text="Sorry, I cannot help.")
        ):
            result = checker._patch(SCRIPT_UNMATCHED, SUSPICIOUS_UNMATCHED, "irrelevant_corrected")
        assert result == SCRIPT_UNMATCHED

    def test_repair_no_longer_raises_valueerror(self) -> None:
        """``repair()`` exhausts all tiers gracefully — never raises ValueError."""
        checker = DataLeakageCheckerAgent(model="test")
        with checker.repair_agent.override(
            model=TestModel(custom_output_text="```python\nx = 1\n```")
        ):
            # Previously this would raise ValueError("not found").
            # Now it should return the original code or the rewritten code.
            result = checker.repair(SCRIPT_UNMATCHED, SUSPICIOUS_UNMATCHED)
        assert isinstance(result, str)
        # Should not raise — that's the test
