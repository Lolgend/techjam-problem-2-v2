# Implementation Plan: Fix Subprocess Sandbox Path Resolution in SubprocessRunner

## Phase 1: Absolute Path Normalization in SubprocessRunner [checkpoint: ddcb23c]
- [x] Task: Write tests for relative runs_dir path execution in `tests/runner/test_sandbox.py`
    - [x] Add unit test verifying that SubprocessRunner with relative `runs_dir` executes `solution.py` without `[Errno 2]`
- [x] Task: Implement absolute path normalization in `src/problem_2_v2/runner/sandbox.py`
    - [x] Ensure `prepare_sandbox` and `run_code` resolve `sandbox` and `script_path` to absolute paths via `.resolve()`
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Full System Integration & Regression Verification [checkpoint: 40ca758]
- [x] Task: Run full test suite and verify 100% pass rate
    - [x] Execute `uv run pytest --tb=short -q` across all 310+ tests (348 passed)
- [~] Task: Verify candidate execution in real sandbox with relative `runs` directory
    - [x] Run test execution verifying candidate training script execution
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)
