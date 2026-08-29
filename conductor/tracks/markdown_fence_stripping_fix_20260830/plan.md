# Implementation Plan: Fix Markdown Fence Stripping in Debugger & Guardrails

## Phase 1: Markdown Fence Stripping in Debugger, Guardrails & Code Extractor
- [ ] Task: Write tests for markdown fence stripping in debugger and code extractor
    - [ ] Add unit tests in `tests/runner/test_debugger.py` and `tests/test_code_utils.py` verifying fence removal
- [ ] Task: Implement fence stripping across debugger.py, leakage.py, and code_utils.py
    - [ ] Add `extract_python_code()` in `DebuggerAgent.debug()`
    - [ ] Add `extract_python_code()` in `DataLeakageCheckerAgent.repair()`
    - [ ] Fortify `extract_python_code()` in `src/problem_2_v2/contracts/code_utils.py`
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Full System Integration & Regression Verification
- [ ] Task: Run full test suite and verify 100% pass rate
    - [ ] Execute `uv run pytest --tb=short -q` across all 310+ tests
- [ ] Task: Verify candidate execution in real sandbox with candidate models
    - [ ] Run test execution verifying candidate training script execution without SyntaxError
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)
