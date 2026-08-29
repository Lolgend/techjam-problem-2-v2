"""Unit tests for the code helper utilities in `contracts.code_utils`."""

from problem_2_v2.contracts.code_utils import (
    compute_code_diff,
    extract_python_code,
    validate_python_syntax,
)


class TestExtractPythonCode:
    """Test `extract_python_code` markdown-aware code extraction."""

    def test_extracts_fenced_python_block(self) -> None:
        """A ```python fenced block should have its fences and noise stripped."""
        text = (
            "Here is the solution:\n```python\nimport numpy as np\n\n"
            "result = np.mean([1, 2, 3])\n```\nDone."
        )
        expected = "import numpy as np\n\nresult = np.mean([1, 2, 3])"
        assert extract_python_code(text) == expected

    def test_extracts_bare_fenced_block(self) -> None:
        """A fence without a language tag should still be stripped."""
        text = '```\nprint("hello")\n```'
        assert extract_python_code(text) == 'print("hello")'

    def test_preserves_raw_code(self) -> None:
        """Raw code without any markdown fences should be returned stripped."""
        raw = "\n\nimport os\n\npath = os.getcwd()\n\n"
        assert extract_python_code(raw) == "import os\n\npath = os.getcwd()"

    def test_extracts_first_block_from_mixed_text(self) -> None:
        """When prose contains multiple fences, the first python block wins."""
        text = "Some prose.\n```python\nx = 1\n```\nMore prose.\n```python\ny = 2\n```"
        assert extract_python_code(text) == "x = 1"

    def test_returns_empty_string_when_no_code(self) -> None:
        """Prose without any code should produce an empty string."""
        assert extract_python_code("Just some plain prose without code.") == ""
        assert extract_python_code("") == ""

    def test_strips_crlf_line_endings(self) -> None:
        """Windows-style line endings inside fenced blocks should be normalized."""
        text = '```python\r\nprint("a")\r\n```'
        assert extract_python_code(text) == 'print("a")'

    def test_strips_leading_fence_from_unclosed_block(self) -> None:
        """A response with an opening fence but no closing fence is recovered."""
        text = "```python\nimport numpy as np\nx = np.array([1])\n"
        assert extract_python_code(text) == "import numpy as np\nx = np.array([1])"

    def test_strips_residual_trailing_fence(self) -> None:
        """A stray trailing fence line is removed before parsing."""
        text = "print('x')\n```\n"
        assert extract_python_code(text) == "print('x')"

    def test_strips_residual_fences_around_fenced_block(self) -> None:
        """Well-formed fenced responses are still extracted cleanly."""
        text = "```python\nprint('ok')\n```\n"
        assert extract_python_code(text) == "print('ok')"


class TestValidatePythonSyntax:
    """Test `validate_python_syntax` AST-based syntax validation."""

    def test_valid_code(self) -> None:
        valid, error = validate_python_syntax("def add(a: int, b: int) -> int:\n    return a + b\n")
        assert valid is True
        assert error is None

    def test_syntax_error(self) -> None:
        valid, error = validate_python_syntax("def broken(:\n    pass")
        assert valid is False
        assert error is not None
        assert "invalid syntax" in error

    def test_empty_code_is_valid(self) -> None:
        valid, error = validate_python_syntax("")
        assert valid is True
        assert error is None

    def test_error_reports_line_number(self) -> None:
        valid, error = validate_python_syntax('x = 1\nif True print("bad")')
        assert valid is False
        assert error is not None
        assert "line 2" in error


class TestComputeCodeDiff:
    """Test `compute_code_diff` unified diff generation."""

    def test_produces_unified_diff(self) -> None:
        old_code = "x = 1\n"
        new_code = "x = 2\n"
        diff = compute_code_diff(old_code, new_code)
        assert "---" in diff
        assert "+++" in diff
        assert "-x = 1" in diff
        assert "+x = 2" in diff

    def test_identical_code_produces_empty_diff(self) -> None:
        code = "a = 1\nb = 2\n"
        assert compute_code_diff(code, code) == ""

    def test_added_lines_are_prefixed_with_plus(self) -> None:
        old_code = "a = 1\n"
        new_code = "a = 1\nb = 2\n"
        diff = compute_code_diff(old_code, new_code)
        assert "+b = 2" in diff
        assert " a = 1" in diff

    def test_removed_lines_are_prefixed_with_minus(self) -> None:
        old_code = "a = 1\nb = 2\n"
        new_code = "a = 1\n"
        diff = compute_code_diff(old_code, new_code)
        assert "-b = 2" in diff

    def test_diff_includes_lineno_hunk_header(self) -> None:
        old_code = "a = 1\nb = 2\nc = 3\n"
        new_code = "a = 1\nb = 20\nc = 3\n"
        diff = compute_code_diff(old_code, new_code)
        assert diff.startswith("@@") or "\n@@" in diff
