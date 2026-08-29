"""Code helper utilities for the MLE-STAR contract layer.

This module provides low-level helpers for cleaning LLM-generated Python
code, validating syntax via the standard library AST module, and computing
unified diffs for iteration logging.
"""

from __future__ import annotations

import ast
import difflib
import re

_FENCED_BLOCK_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)


def extract_python_code(text: str) -> str:
    """Extract clean Python source from an LLM markdown response.

    Args:
        text: Raw model output, which may contain fenced code blocks
            (`````python ... `````), bare fences (````` ... `````), or raw
            code mixed with prose.

    Returns:
        The cleaned Python source. When the text contains fenced blocks,
        the first block is returned. Otherwise the text is stripped of
        leading/trailing whitespace and returned when it parses as valid
        Python (raw code); prose that is not valid Python yields an empty
        string.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    match = _FENCED_BLOCK_RE.search(text)
    if match is not None:
        return match.group(1).strip()
    stripped = text.strip()
    if stripped and _parses_as_python(stripped):
        return stripped
    return ""


def _parses_as_python(code: str) -> bool:
    """Return whether ``code`` parses as valid Python syntax."""
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def validate_python_syntax(code: str) -> tuple[bool, str | None]:
    """Validate Python source code with the standard library AST parser.

    Args:
        code: Python source to validate. May be empty.

    Returns:
        A tuple of ``(is_valid, error_message)``. When the code is valid the
        error message is ``None``; otherwise it is a human-readable
        ``SyntaxError`` message including the offending line number.
    """
    try:
        ast.parse(code)
    except SyntaxError as exc:
        message = exc.msg or "invalid syntax"
        line_info = f"line {exc.lineno}" if exc.lineno is not None else "unknown line"
        return False, f"{message} ({line_info})"
    return True, None


def compute_code_diff(old_code: str, new_code: str) -> str:
    """Compute a clean unified diff between two versions of source code.

    Args:
        old_code: The original source.
        new_code: The modified source.

    Returns:
        A unified diff string (with ``---``/``+++`` file headers and hunk
        headers). Returns an empty string when the two inputs are
        identical.
    """
    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile="old_code",
        tofile="new_code",
        lineterm="\n",
    )
    return "".join(diff)
