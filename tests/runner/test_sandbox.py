"""Unit tests for the isolated subprocess execution sandbox."""

import sys
import time
from pathlib import Path

import pytest

from problem_2_v2.contracts.task import ExecutionResult
from problem_2_v2.runner.sandbox import SubprocessRunner

OK_SCRIPT = "print('Final Validation Performance: 0.9123')"
FAILING_SCRIPT = "import sys\nprint('about to fail', flush=True)\nsys.exit(3)"
SYNTAX_ERROR_SCRIPT = "def broken(:\n    pass\n"
EXCEPTION_SCRIPT = "raise ValueError('boom')"
NO_SCORE_SCRIPT = "print('no score here')"
SLOW_SCRIPT = "import time\ntime.sleep(30)\nprint('done')"


@pytest.fixture()
def runner(tmp_path: Path) -> SubprocessRunner:
    return SubprocessRunner(
        runs_dir=str(tmp_path / "runs"),
        timeout_seconds=5,
        python_executable=sys.executable,
    )


class TestSandboxWorkspace:
    """Test isolated sandbox directory creation and input mapping."""

    def test_creates_sandbox_directory(self, runner: SubprocessRunner) -> None:
        sandbox = runner.prepare_sandbox(run_id="run1", candidate_id="cand1")
        assert sandbox.exists()
        assert sandbox.name == "sandbox_cand1"
        assert (runner.runs_dir / "run1").exists()

    def test_input_files_are_mapped(self, runner: SubprocessRunner, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        (data_dir / "test.csv").write_text("a,b\n3,4\n", encoding="utf-8")

        sandbox = runner.prepare_sandbox(
            run_id="run1",
            candidate_id="cand1",
            dataset_dir=str(data_dir),
            dataset_files=["train.csv", "test.csv"],
        )
        assert (sandbox / "input" / "train.csv").read_text(encoding="utf-8") == "a,b\n1,2\n"
        assert (sandbox / "input" / "test.csv").read_text(encoding="utf-8") == "a,b\n3,4\n"


class TestSandboxBaselineAccess:
    """Test PYTHONPATH injection for the official baseline helper modules."""

    def test_sandbox_scripts_can_import_baseline_modules(
        self, runner: SubprocessRunner
    ) -> None:
        code = (
            "import evaluate\n"
            "import data\n"
            "import submit\n"
            "from evaluate import evaluate as eval_fn\n"
            "print('Final Validation Performance: 0.5')\n"
        )
        sandbox = runner.prepare_sandbox(run_id="r", candidate_id="c")
        result = runner.run_code(code, sandbox_dir=str(sandbox))
        assert result.success is True
        assert result.returncode == 0

    def test_run_code_injects_workspace_and_baseline_on_pythonpath(
        self, runner: SubprocessRunner
    ) -> None:
        baseline = str(runner.baseline_dir.resolve())
        workspace = str(runner.workspace_root.resolve())
        code = (
            "import sys\n"
            f"baseline = r'{baseline}'\n"
            f"workspace = r'{workspace}'\n"
            "paths = [p.casefold() for p in sys.path]\n"
            "assert baseline.casefold() in paths, paths\n"
            "assert workspace.casefold() in paths, paths\n"
            "print('Final Validation Performance: 0.5')\n"
        )
        sandbox = runner.prepare_sandbox(run_id="r", candidate_id="c")
        result = runner.run_code(code, sandbox_dir=str(sandbox))
        assert result.success is True
        assert result.returncode == 0

    def test_baseline_dirs_default_to_package_layout(self) -> None:
        runner = SubprocessRunner(timeout_seconds=5, python_executable=sys.executable)
        assert runner.baseline_dir.is_absolute()
        assert (runner.baseline_dir / "evaluate.py").is_file()
        assert (runner.baseline_dir / "submit.py").is_file()
        assert (runner.workspace_root / "src" / "baseline").resolve() == runner.baseline_dir

    def test_explicit_baseline_dirs_are_honored(self, tmp_path: Path) -> None:
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        workspace = tmp_path / "repo"
        workspace.mkdir()
        runner = SubprocessRunner(
            runs_dir=str(tmp_path / "runs"),
            timeout_seconds=5,
            python_executable=sys.executable,
            workspace_root=str(workspace),
            baseline_dir=str(baseline),
        )
        assert runner.workspace_root == workspace.resolve()
        assert runner.baseline_dir == baseline.resolve()


class TestSubprocessExecution:
    """Test execution capture, timeout, and error handling."""

    def test_successful_run_captures_score(self, runner: SubprocessRunner) -> None:
        sandbox = runner.prepare_sandbox(run_id="r", candidate_id="c")
        result = runner.run_code(OK_SCRIPT, sandbox_dir=str(sandbox))
        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.returncode == 0
        assert result.validation_score == pytest.approx(0.9123)
        assert "Final Validation Performance: 0.9123" in result.stdout

    def test_relative_runs_dir_executes_cleanly(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        relative_runner = SubprocessRunner(
            runs_dir="relative_runs",
            timeout_seconds=5,
            python_executable=sys.executable,
        )
        sandbox = relative_runner.prepare_sandbox(run_id="run1", candidate_id="cand1")
        assert sandbox.is_absolute()
        assert "relative_runs" in str(sandbox)
        result = relative_runner.run_code(OK_SCRIPT, sandbox_dir=str(sandbox))
        assert result.success is True
        assert result.returncode == 0
        assert result.validation_score == pytest.approx(0.9123)
        assert (sandbox / "solution.py").exists()

    def test_run_code_resolves_relative_sandbox_dir(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        relative_runner = SubprocessRunner(
            runs_dir="relative_runs",
            timeout_seconds=5,
            python_executable=sys.executable,
        )
        relative_runner.prepare_sandbox(run_id="run1", candidate_id="cand1")
        relative_sandbox = Path("relative_runs") / "run1" / "sandbox_cand1"
        result = relative_runner.run_code(OK_SCRIPT, sandbox_dir=str(relative_sandbox))
        assert result.success is True
        assert result.validation_score == pytest.approx(0.9123)

    def test_non_zero_exit_is_failure(self, runner: SubprocessRunner) -> None:
        sandbox = runner.prepare_sandbox(run_id="r", candidate_id="c")
        result = runner.run_code(FAILING_SCRIPT, sandbox_dir=str(sandbox))
        assert result.success is False
        assert result.returncode == 3
        assert "about to fail" in result.stdout

    def test_syntax_error_is_captured(self, runner: SubprocessRunner) -> None:
        sandbox = runner.prepare_sandbox(run_id="r", candidate_id="c")
        result = runner.run_code(SYNTAX_ERROR_SCRIPT, sandbox_dir=str(sandbox))
        assert result.success is False
        assert result.returncode != 0
        assert "SyntaxError" in result.stderr or "invalid syntax" in result.stderr

    def test_runtime_exception_traceback_is_captured(self, runner: SubprocessRunner) -> None:
        sandbox = runner.prepare_sandbox(run_id="r", candidate_id="c")
        result = runner.run_code(EXCEPTION_SCRIPT, sandbox_dir=str(sandbox))
        assert result.success is False
        assert "Traceback" in result.stderr
        assert "ValueError: boom" in result.stderr

    def test_missing_score_line_marks_success_but_no_score(self, runner: SubprocessRunner) -> None:
        sandbox = runner.prepare_sandbox(run_id="r", candidate_id="c")
        result = runner.run_code(NO_SCORE_SCRIPT, sandbox_dir=str(sandbox))
        assert result.success is True
        assert result.returncode == 0
        assert result.validation_score is None

    def test_timeout_terminates_and_marks_failure(self, runner: SubprocessRunner) -> None:
        sandbox = runner.prepare_sandbox(run_id="r", candidate_id="c")
        started = time.monotonic()
        result = runner.run_code(SLOW_SCRIPT, sandbox_dir=str(sandbox))
        elapsed = time.monotonic() - started
        assert elapsed < 15
        assert result.success is False
        assert result.returncode != 0
        assert result.stderr or result.stdout

    def test_script_is_written_into_sandbox(self, runner: SubprocessRunner) -> None:
        sandbox = runner.prepare_sandbox(run_id="r", candidate_id="c")
        result = runner.run_code("x = 1\n", sandbox_dir=str(sandbox))
        assert result is not None
        assert (sandbox / "solution.py").exists()

    def test_duration_is_measured(self, runner: SubprocessRunner) -> None:
        sandbox = runner.prepare_sandbox(run_id="r", candidate_id="c")
        result = runner.run_code(OK_SCRIPT, sandbox_dir=str(sandbox))
        assert result.duration_seconds >= 0.0
