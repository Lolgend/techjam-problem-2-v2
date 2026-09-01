"""Code helper utilities for the MLE-STAR contract layer.

This module provides low-level helpers for cleaning LLM-generated Python
code, validating syntax via the standard library AST module, and computing
unified diffs for iteration logging.
"""

from __future__ import annotations

import ast
import difflib
import re
from typing import Any

_FENCED_BLOCK_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)
_FENCE_LINE_RE = re.compile(r"^\s*```[^\n]*$")


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
        string. Residual leading/trailing markdown fence lines are removed
        unconditionally so malformed LLM responses never leak `` ``` ``
        markers into executed code.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    match = _FENCED_BLOCK_RE.search(text)
    if match is not None:
        return _strip_residual_fences(match.group(1)).strip()
    stripped = text.strip()
    if stripped and _parses_as_python(stripped):
        return stripped
    cleaned = _strip_residual_fences(stripped)
    if cleaned and _parses_as_python(cleaned):
        return cleaned.strip()
    return ""


def _strip_residual_fences(text: str) -> str:
    """Remove any leading/trailing markdown backtick fence lines.

    Args:
        text: Text that may begin or end with backtick fence lines.

    Returns:
        The text with leading and trailing fence lines removed.
    """
    lines = text.splitlines()
    while lines and _FENCE_LINE_RE.match(lines[0]):
        lines.pop(0)
    while lines and _FENCE_LINE_RE.match(lines[-1]):
        lines.pop()
    return "\n".join(lines)


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


def is_truncated_code(raw_text: str, extracted_code: str) -> bool:
    """Check whether an LLM response appears truncated by a max_tokens cutoff.

    Signs of truncation:
    1. An opening code fence exists without a matching closing fence.
    2. Syntax validation errors indicating unexpected EOF / unclosed brackets.

    Args:
        raw_text: The full raw output string from the LLM.
        extracted_code: The parsed python code extracted from the output.

    Returns:
        True if the output appears prematurely truncated, False otherwise.
    """
    cleaned_raw = raw_text.strip()
    if not cleaned_raw:
        return False
    # Check for unclosed markdown code fence (e.g. ```python without closing ```)
    if "```" in cleaned_raw and cleaned_raw.count("```") % 2 != 0:
        return True

    # Check for syntax errors caused by premature EOF
    if extracted_code:
        valid, error = validate_python_syntax(extracted_code)
        if not valid and error:
            lower_err = error.lower()
            if (
                "unexpected eof" in lower_err
                or "eof while parsing" in lower_err
                or "unclosed" in lower_err
                or "was never closed" in lower_err
            ):
                return True
    return False


def run_agent_sync_safe(
    agent: Any,
    prompt: str,
    max_retries: int = 2,
) -> Any:
    """Run a Pydantic AI agent synchronously with automatic recovery from UnexpectedModelBehavior.

    Handles cases where thinking/reasoning models exhaust the token budget before
    emitting content by re-prompting with a direct code generation instruction.

    Args:
        agent: The Pydantic AI agent to execute.
        prompt: The initial prompt string.
        max_retries: Maximum number of retries upon token limit errors.

    Returns:
        The agent's RunResult.
    """
    import logfire

    current_prompt = prompt
    for attempt in range(max_retries + 1):
        try:
            return agent.run_sync(current_prompt)
        except Exception as exc:
            exc_str = str(exc)
            if "Model token limit" in exc_str or "before any response was generated" in exc_str:
                if attempt < max_retries:
                    logfire.warn("agent.token_limit_retry", attempt=attempt, error=exc_str)
                    current_prompt = (
                        f"{prompt}\n\n"
                        "# CRITICAL INSTRUCTION (REASONING BUDGET RECOVERY)\n"
                        "Your previous response exceeded the token budget during reasoning before generating any output.\n"
                        "Please output the complete Python code block immediately with minimal reasoning wrapped in ```python ... ```."
                    )
                    continue
            raise
