"""Isolated subprocess execution sandbox with telemetry capture.

Runs generated Python scripts in per-candidate scratch directories with a
``./input`` mapping to the dataset, a configurable timeout, and full
stdout/stderr/score telemetry.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import logfire

from problem_2_v2.contracts.task import ExecutionResult


def _resolve_workspace_root() -> Path:
    """Resolve the repository root containing ``src/baseline``.

    Walks upward from this package until a directory holding the official
    baseline modules is found, falling back to a fixed package-depth guess
    so the default works under non-editable installs.
    """
    start = Path(__file__).resolve()
    for parent in (start.parent, *start.parent.parents):
        if (parent / "src" / "baseline" / "evaluate.py").is_file():
            return parent
    return Path(__file__).resolve().parents[3]


class SubprocessRunner:
    """Executes Python scripts in isolated sandbox directories.

    Attributes:
        runs_dir: Root directory holding per-run sandboxes
            (``runs/<run_id>/sandbox_<candidate_id>/``).
        timeout_seconds: Per-script wall-clock timeout.
        python_executable: Python interpreter used to run scripts.
        cuda_devices: Comma-separated CUDA device ids exported to the
            sandbox process, if any.
        workspace_root: Project root injected into the subprocess
            ``PYTHONPATH``.
        baseline_dir: Directory holding the official baseline helper
            modules (``evaluate.py``, ``submit.py``, ``data.py``), also
            injected into the subprocess ``PYTHONPATH``.
    """

    def __init__(
        self,
        runs_dir: str = "runs",
        timeout_seconds: int = 1200,
        python_executable: str | None = None,
        cuda_devices: str | None = None,
        workspace_root: str | None = None,
        baseline_dir: str | None = None,
    ) -> None:
        """Create a subprocess runner.

        Args:
            runs_dir: Root directory for per-run sandboxes.
            timeout_seconds: Per-script timeout in seconds (default 600).
            python_executable: Interpreter to use; defaults to the current
                interpreter.
            cuda_devices: Optional ``CUDA_VISIBLE_DEVICES`` value.
            workspace_root: Project root to inject into the subprocess
                ``PYTHONPATH``; defaults to the repository root resolved
                from this package's location.
            baseline_dir: Directory holding the baseline helper modules;
                defaults to ``<workspace_root>/src/baseline``.
        """
        self.runs_dir = Path(runs_dir)
        self.timeout_seconds = timeout_seconds
        self.python_executable = python_executable or sys.executable
        self.cuda_devices = cuda_devices
        self.workspace_root = (
            Path(workspace_root).resolve()
            if workspace_root is not None
            else _resolve_workspace_root()
        )
        self.baseline_dir = (
            Path(baseline_dir).resolve()
            if baseline_dir is not None
            else (self.workspace_root / "src" / "baseline").resolve()
        )

    def prepare_sandbox(
        self,
        run_id: str,
        candidate_id: str,
        dataset_dir: str | None = None,
        dataset_files: list[str] | None = None,
    ) -> Path:
        """Create an isolated sandbox directory and map input data.

        Args:
            run_id: Identifier of the pipeline run.
            candidate_id: Identifier of the candidate script.
            dataset_dir: Directory containing the dataset files.
            dataset_files: File names to map into ``./input``.

        Returns:
            The path to the prepared sandbox directory.
        """
        sandbox = (self.runs_dir / run_id / f"sandbox_{candidate_id}").resolve()
        sandbox.mkdir(parents=True, exist_ok=True)
        input_dir = sandbox / "input"
        input_dir.mkdir(exist_ok=True)

        if dataset_dir is not None:
            dataset_path = Path(dataset_dir)
            for name in dataset_files or []:
                source = dataset_path / name
                if not source.exists():
                    continue
                target = input_dir / name
                if target.exists():
                    try:
                        if source.samefile(target):
                            continue
                    except OSError:
                        pass
                    with contextlib.suppress(OSError):
                        target.unlink()
                try:
                    os.link(source, target)
                except shutil.SameFileError:
                    shutil.copy2(source, target)
                except FileExistsError:
                    shutil.copy2(source, target)
                except OSError:
                    shutil.copy2(source, target)
        return sandbox

    def run_code(
        self, code: str, sandbox_dir: str, script_name: str = "solution.py"
    ) -> ExecutionResult:
        """Write code into the sandbox and execute it as a subprocess.

        Args:
            code: The self-contained Python script to execute.
            sandbox_dir: Prepared sandbox directory for this candidate.
            script_name: Name of the script file inside the sandbox
                (default: ``solution.py``).

        Returns:
            An ``ExecutionResult`` with stdout, stderr, return code,
            duration, and the parsed validation score.
        """
        sandbox = Path(sandbox_dir).resolve()
        script_path = (sandbox / script_name).resolve()
        script_path.write_text(code, encoding="utf-8")

        env = os.environ.copy()
        if self.cuda_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_devices
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        pythonpath = env.get("PYTHONPATH", "").split(os.pathsep)
        pythonpath = [p for p in pythonpath if p]
        pythonpath.extend(
            [
                str(self.workspace_root),
                str(self.baseline_dir),
            ]
        )
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)

        started = time.monotonic()
        try:
            with logfire.span("runner.execute", sandbox=str(sandbox)):
                # S603: intentional — LLM-generated code runs in an isolated
                # sandbox cwd with list-form args (no shell) and a timeout.
                completed = subprocess.run(  # noqa: S603
                    [self.python_executable, str(script_path)],
                    cwd=str(sandbox),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - started
            logfire.warn("runner.execute.timed_out", timeout_seconds=self.timeout_seconds)
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {self.timeout_seconds}s.",
                returncode=-1,
                duration_seconds=duration,
            )
        duration = time.monotonic() - started

        result = ExecutionResult(
            success=completed.returncode == 0,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            returncode=completed.returncode,
            duration_seconds=duration,
        )
        score = result.extract_validation_score(result.stdout)
        result.validation_score = score

        from problem_2_v2.console import is_verbose

        if is_verbose():
            if result.stdout.strip():
                header = f"--- [Sandbox Output ({sandbox.name})] ---"
                print(f"\n{header}\n{result.stdout.strip()}\n{'=' * len(header)}", flush=True)
            if result.stderr.strip():
                header = f"--- [Sandbox Stderr ({sandbox.name})] ---"
                print(f"\n{header}\n{result.stderr.strip()}\n{'=' * len(header)}", flush=True)

        return result
