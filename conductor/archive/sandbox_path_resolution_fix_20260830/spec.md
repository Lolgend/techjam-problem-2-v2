# Specification: Fix Subprocess Sandbox Path Resolution in SubprocessRunner

## 1. Overview
Fixes the duplicate relative path resolution bug in `SubprocessRunner` where `subprocess.run` with `cwd=sandbox` attempted to execute a relative `script_path` inside the sandbox working directory, causing `[Errno 2] No such file or directory` during candidate model training.

## 2. Functional Requirements

### A. Absolute Path Normalization (`src/problem_2_v2/runner/sandbox.py`)
- In `prepare_sandbox()`: Ensure `sandbox = (self.runs_dir / run_id / f"sandbox_{candidate_id}").resolve()`.
- In `run_code()`: Ensure `sandbox = Path(sandbox_dir).resolve()` and `script_path = (sandbox / "solution.py").resolve()`.
- Pass `[self.python_executable, str(script_path)]` with `cwd=str(sandbox)` to `subprocess.run`.

### B. Path Resolution Unit Tests (`tests/runner/test_sandbox.py`)
- Add unit test verifying that `SubprocessRunner(runs_dir="relative_runs")` creates and executes `solution.py` without duplicate path nesting or `[Errno 2]`.

## 3. Non-Functional Requirements
- **Platform Portability:** Handles Windows backslash paths, Linux forward slashes, and relative paths identically.
- **Zero Regressions:** All existing 310 tests continue to pass 100% green.

## 4. Acceptance Criteria
- [ ] Subprocess execution with relative `runs_dir` executes `solution.py` cleanly without `[Errno 2]`.
- [ ] Sandbox returns valid `ExecutionResult` with stdout, stderr, and parsed validation score.
- [ ] Full test suite passes 100% green.
