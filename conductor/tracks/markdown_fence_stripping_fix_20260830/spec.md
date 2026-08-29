# Specification: Fix Markdown Fence Stripping in Debugger & Guardrails

## 1. Overview
Fixes the issue where `DebuggerAgent` and `DataLeakageCheckerAgent` failed to strip markdown code fences from LLM responses, writing ````python` into `solution.py` and causing `SyntaxError: invalid syntax` on line 1 during candidate model training.

## 2. Functional Requirements

### A. Debugger Code Extraction (`src/problem_2_v2/runner/debugger.py`)
- Wrap `response.output` with `extract_python_code()` in `DebuggerAgent.debug()` before re-executing in the sandbox.

### B. Guardrails Code Extraction (`src/problem_2_v2/guardrails/leakage.py`)
- Wrap `response.output` with `extract_python_code()` in `DataLeakageCheckerAgent.repair()`.

### C. Fortified Code Extractor (`src/problem_2_v2/contracts/code_utils.py`)
- Update `extract_python_code()` to unconditionally strip any residual leading/trailing backtick lines (` ``` ` or ` ```python `).

### D. Unit Test Verification
- Add unit tests in `tests/runner/test_debugger.py` and `tests/test_code_utils.py` verifying markdown fence stripping during debugger repair rounds.

## 3. Non-Functional Requirements
- **Robustness:** Guaranteed zero syntax errors resulting from LLM markdown code blocks.
- **Zero Regressions:** All existing 310 tests continue to pass 100% green.

## 4. Acceptance Criteria
- [ ] Debugger repair outputs are stripped of markdown fences before sandbox execution.
- [ ] All 5 candidate models run without `SyntaxError: invalid syntax`.
- [ ] Full test suite passes 100% green.
