"""Command-line interface for the ``problem-2-v2`` package.

Provides the ``run`` subcommand (full pipeline or ``--dry-run``) and the
``version`` subcommand, wired to the package console script.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from problem_2_v2.config import MLEStarConfig
from problem_2_v2.console import announce, format_delta, format_score
from problem_2_v2.contracts.task import TaskSpecification
from problem_2_v2.orchestrator import MLEStarPipeline, MLEStarResult, configure_event_loop_policy

_PROVIDERS = ("duckduckgo", "tavily", "google", "mock")
_BANNER_WIDTH = 78
_SUBMIT_CHECK_TIMEOUT = 600


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``problem-2-v2`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="problem-2-v2",
        description="Autonomous machine learning engineering agent (MLE-STAR).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run the full MLE-STAR pipeline from markdown to artifacts."
    )
    run_parser.add_argument(
        "--task", "-t", required=True, help="Path to the problem markdown file."
    )
    run_parser.add_argument("--data", "-d", required=True, help="Path to the dataset directory.")
    run_parser.add_argument(
        "--output", "-o", default="final", help="Output directory (default: ./final)."
    )
    run_parser.add_argument("--model", "-m", default="openai:gpt-4o", help="LLM model identifier.")
    run_parser.add_argument(
        "--search-provider",
        "-s",
        choices=_PROVIDERS,
        default="duckduckgo",
        help="Search backend.",
    )
    run_parser.add_argument("--branches", "-b", type=int, default=2, help="Parallel branches (L).")
    run_parser.add_argument(
        "--outer-loops", "-T", type=int, default=3, help="Outer iterations (T)."
    )
    run_parser.add_argument(
        "--inner-loops", "-K", type=int, default=3, help="Inner iterations (K)."
    )
    run_parser.add_argument(
        "--ensemble-rounds", "-R", type=int, default=3, help="Ensemble rounds (R)."
    )
    run_parser.add_argument(
        "--seeds", default=None, help="Comma-separated random seeds per branch."
    )
    run_parser.add_argument(
        "--api-key",
        "-k",
        default=None,
        help=(
            "LLM API key (automatically populates OPENAI_API_KEY, "
            "DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY based on --model)."
        ),
    )
    run_parser.add_argument(
        "--base-url",
        default=None,
        help="Custom API base URL (e.g. https://api.deepseek.com or https://openrouter.ai/api/v1).",
    )
    run_parser.add_argument(
        "--search-api-key",
        default=None,
        help="Search API key (e.g. for Tavily or Google Custom Search).",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose telemetry and live subprocess execution logging.",
    )
    run_parser.add_argument(
        "--logfire-token",
        default=None,
        help="Logfire write token for streaming traces to the web dashboard.",
    )
    run_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum output tokens for LLM responses (default: provider limit).",
    )
    run_parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM sampling temperature (default: provider default).",
    )
    run_parser.add_argument(
        "--thinking",
        choices=["minimal", "low", "medium", "high", "xhigh", "off"],
        default=None,
        help="Thinking effort level for reasoning models like Gemini 2.0/3.7 Thinking or o1/o3-mini ('minimal', 'low', 'medium', 'high', 'xhigh', 'off').",
    )
    run_parser.add_argument(
        "--dry-run", action="store_true", help="Validate inputs without running the pipeline."
    )

    finalize_parser = subparsers.add_parser(
        "finalize", help="Finalize an existing model script into a production artifact."
    )
    finalize_parser.add_argument(
        "--script", "-s", required=True, help="Path to the Python solution script to finalize."
    )
    finalize_parser.add_argument(
        "--task", "-t", required=True, help="Path to the problem markdown file."
    )
    finalize_parser.add_argument(
        "--data", "-d", required=True, help="Path to the dataset directory."
    )
    finalize_parser.add_argument(
        "--output", "-o", default="final", help="Output directory (default: ./final)."
    )
    finalize_parser.add_argument(
        "--model", "-m", default="openai:gpt-4o", help="LLM model identifier for finalizer agent."
    )
    finalize_parser.add_argument(
        "--api-key", default=None, help="LLM provider API key."
    )
    finalize_parser.add_argument(
        "--base-url", default=None, help="Custom base URL for the LLM API."
    )
    finalize_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum output tokens for LLM responses (default: provider limit).",
    )
    finalize_parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM sampling temperature (default: provider default).",
    )
    finalize_parser.add_argument(
        "--thinking",
        choices=["minimal", "low", "medium", "high", "xhigh", "off"],
        default=None,
        help="Thinking effort level for reasoning models ('minimal', 'low', 'medium', 'high', 'xhigh', 'off').",
    )
    finalize_parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Production execution timeout in seconds (default: 3600).",
    )
    finalize_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose telemetry and live subprocess execution logging.",
    )
    finalize_parser.add_argument(
        "--dry-run", action="store_true", help="Validate inputs without running finalization."
    )

    subparsers.add_parser("version", help="Show the version and system info.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return the process exit code.

    Args:
        argv: Optional argument list; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` on a failed run.
    """
    import os

    os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")
    configure_event_loop_policy()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        return _version_command()
    if args.command == "finalize":
        return _finalize_command(args, parser)
    return _run_command(args, parser)


def _version_command() -> int:
    """Print the version and system information."""
    try:
        version = metadata.version("problem-2-v2")
    except metadata.PackageNotFoundError:
        version = "0.1.0"
    print(f"problem-2-v2 {version}")
    print(f"Python {sys.version.split()[0]}")
    print(f"Platform {sys.platform}")
    return 0


def _run_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Execute the ``run`` subcommand (dry-run or full pipeline)."""
    import os

    if args.verbose:
        from problem_2_v2.console import set_verbose

        set_verbose(True)

    if args.logfire_token:
        os.environ["LOGFIRE_TOKEN"] = args.logfire_token

    try:
        import logfire

        logfire.configure(
            service_name="mle-star",
            send_to_logfire="if-token-present",
            token=os.environ.get("LOGFIRE_TOKEN"),
        )
        logfire.instrument_pydantic_ai()
    except Exception as exc:
        print(f"Warning: logfire telemetry unavailable: {exc}", file=sys.stderr)

    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url
        os.environ["DEEPSEEK_BASE_URL"] = args.base_url

    if args.api_key:
        model_str = args.model.lower()
        if model_str.startswith("deepseek:") or "deepseek" in model_str:
            os.environ["DEEPSEEK_API_KEY"] = args.api_key
            os.environ["OPENAI_API_KEY"] = args.api_key
        elif model_str.startswith("anthropic:") or "claude" in model_str:
            os.environ["ANTHROPIC_API_KEY"] = args.api_key
        elif model_str.startswith("google") or "gemini" in model_str:
            os.environ["GEMINI_API_KEY"] = args.api_key
            os.environ["GOOGLE_API_KEY"] = args.api_key
        elif model_str.startswith("openrouter:") or "openrouter" in model_str:
            os.environ["OPENROUTER_API_KEY"] = args.api_key
            os.environ["OPENAI_API_KEY"] = args.api_key
        elif model_str.startswith("groq:"):
            os.environ["GROQ_API_KEY"] = args.api_key
        elif model_str.startswith("mistral:"):
            os.environ["MISTRAL_API_KEY"] = args.api_key
        else:
            os.environ["OPENAI_API_KEY"] = args.api_key

    if args.search_api_key:
        provider = args.search_provider.lower()
        if provider == "tavily":
            os.environ["TAVILY_API_KEY"] = args.search_api_key
        elif provider == "google":
            os.environ["GOOGLE_API_KEY"] = args.search_api_key

    seeds = _parse_seeds(args.seeds)
    if seeds is not None and len(seeds) != args.branches:
        parser.error(f"--seeds ({len(seeds)}) must match --branches ({args.branches}).")
    config = MLEStarConfig(
        model=args.model,
        search_provider=args.search_provider,
        num_branches=args.branches,
        outer_loops=args.outer_loops,
        inner_loops=args.inner_loops,
        ensemble_rounds=args.ensemble_rounds,
        seeds=seeds,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        thinking=args.thinking,
    )
    pipeline = MLEStarPipeline(config=config)

    try:
        spec = pipeline.validate(args.task, args.data)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_banner(spec, config)

    if args.dry_run:
        print(f"Dry-run OK: task '{spec.task_name}', baseline {spec.baseline_score:.4f}")
        return 0

    try:
        result = pipeline.run(args.task, args.data)
    except (FileNotFoundError, NotADirectoryError) as exc:
        parser.error(str(exc))
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1

    if result.final_artifact is not None:
        _copy_final_output(result.final_artifact.output_dir, args.output)
    submission_verified, submission_message = _verify_submission(args.output, args.data)
    _print_summary(
        result,
        args.output,
        submission_verified=submission_verified,
        submission_message=submission_message,
    )
    return 0 if result.success else 1


def _finalize_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Execute the standalone finalize workflow on an existing script."""
    import os

    if args.verbose:
        from problem_2_v2.console import set_verbose

        set_verbose(True)

    if args.api_key:
        model_str = args.model.lower()
        if model_str.startswith("deepseek:"):
            os.environ["DEEPSEEK_API_KEY"] = args.api_key
        elif model_str.startswith("google:") or model_str.startswith("google-gla:"):
            os.environ["GEMINI_API_KEY"] = args.api_key
        elif model_str.startswith("openrouter:"):
            os.environ["OPENROUTER_API_KEY"] = args.api_key
        elif model_str.startswith("groq:"):
            os.environ["GROQ_API_KEY"] = args.api_key
        elif model_str.startswith("mistral:"):
            os.environ["MISTRAL_API_KEY"] = args.api_key
        else:
            os.environ["OPENAI_API_KEY"] = args.api_key

    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url

    script_path = Path(args.script)
    if not script_path.is_file():
        parser.error(f"Script file not found: {args.script}")

    code = script_path.read_text(encoding="utf-8")

    config = MLEStarConfig(
        model=args.model,
        production_timeout_seconds=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        thinking=args.thinking,
    )
    pipeline = MLEStarPipeline(config=config)

    try:
        spec = pipeline.validate(args.task, args.data)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"Dry-run OK: script '{script_path.name}', task '{spec.task_name}'")
        return 0

    announce("=" * _BANNER_WIDTH)
    announce("MLE-STAR: Standalone Finalize & Production Artifact Producer")
    announce(f"Script: {script_path.name} | Task: {spec.task_name}")
    announce(f"Dataset: {spec.dataset_dir}")
    announce("=" * _BANNER_WIDTH)

    try:
        artifact = pipeline.finalizer.produce(
            code=code,
            spec=spec,
            run_id="standalone_finalize",
        )
    except Exception as exc:
        print(f"Finalization failed: {exc}", file=sys.stderr)
        return 1

    if artifact.success and artifact.output_dir:
        _copy_final_output(artifact.output_dir, args.output)
        submission_verified, submission_message = _verify_submission(args.output, args.data)
        announce("=" * _BANNER_WIDTH)
        announce(f"Finalization succeeded! Score: {format_score(artifact.validation_score)}")
        announce(f"Output directory: {args.output}")
        announce(f"Submission status: {submission_message}")
        announce("=" * _BANNER_WIDTH)
        return 0

    announce("Finalization completed with errors (no validation score produced).", level="ERROR")
    return 1


def _print_banner(spec: TaskSpecification, config: MLEStarConfig) -> None:
    """Render the startup banner with task and configuration details."""
    announce("=" * _BANNER_WIDTH)
    announce("MLE-STAR: Autonomous Machine Learning Engineering Agent")
    announce(
        f"Task: {spec.task_name} | Type: {spec.task_type.value} | "
        f"Metric: {spec.metric_name} (Baseline: {spec.baseline_score:.4f})"
    )
    announce(f"Dataset: {spec.dataset_dir}")
    announce(
        f"Model: {config.model} | Search: {config.search_provider} | "
        f"Branches: {config.num_branches} | T: {config.outer_loops} K: {config.inner_loops} | "
        f"Ensembles: {config.ensemble_rounds}"
    )
    announce("=" * _BANNER_WIDTH)


def _print_summary(
    result: MLEStarResult,
    output_dir: str,
    submission_verified: bool | None = None,
    submission_message: str | None = None,
) -> None:
    """Render the final summary box with duration, scores, and artifacts."""
    announce("=" * _BANNER_WIDTH)
    announce(f"Run complete in {result.duration_seconds:.1f}s")
    announce(
        f"Baseline: {result.baseline_score:.4f} | "
        f"Final: {format_score(result.final_score)} | Delta: {format_delta(result.score_delta)}"
    )
    if submission_verified is True:
        announce(f"Submission check: PASSED - {submission_message}")
    elif submission_verified is False:
        announce(f"Submission check: FAILED - {submission_message}")
    else:
        announce("Submission check: not applicable")
    artifact_dir = Path(output_dir)
    files = (
        sorted(p.name for p in artifact_dir.iterdir() if p.is_file())
        if artifact_dir.is_dir()
        else []
    )
    if files:
        announce("Artifacts:")
        for name in files:
            announce(f"  - {artifact_dir / name}")
    else:
        announce("Artifacts: (none)")
    announce("=" * _BANNER_WIDTH)


def _parse_seeds(raw: str | None) -> list[int] | None:
    """Parse a comma-separated seed string into a list of integers."""
    if raw is None:
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _copy_final_output(source: str, destination: str) -> None:
    """Copy production artifacts from the sandbox ``./final`` to the output dir."""
    src = Path(source)
    if not src.is_dir():
        return
    dst = Path(destination)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if not item.is_file():
            continue
        target = dst / item.name
        if target.exists() and target.resolve() == item.resolve():
            continue
        shutil.copy2(item, target)


def _submit_script_path() -> Path | None:
    """Resolve the official baseline ``src/baseline/submit.py`` script.

    The script is located by walking upward from this package to the
    repository root, supporting editable and installed layouts.
    """
    start = Path(__file__).resolve()
    for parent in (start, *start.parents):
        candidate = parent / "src" / "baseline" / "submit.py"
        if candidate.is_file():
            return candidate
    return None


def _ascii_safe(text: str) -> str:
    """Return a console-safe ASCII copy of ``text``.

    The baseline ``submit.py`` reports success/errors in non-ASCII text
    (e.g. ``✓`` and Chinese), which cannot be encoded to a cp1252 console.
    Non-ASCII characters are replaced with ``?`` so summaries never crash.
    """
    return text.encode("ascii", "replace").decode("ascii")


def _verify_submission(output_dir: str, data_dir: str) -> tuple[bool | None, str]:
    """Verify a produced ``submission.csv`` with ``submit.py --check``.

    Args:
        output_dir: Directory holding the copied production artifacts.
        data_dir: Dataset directory passed to ``submit.py --data_dir`` so
            row alignment is checked against the official test split.

    Returns:
        A tuple ``(verified, message)`` where ``verified`` is ``True`` when
        the check passes, ``False`` when it fails, and ``None`` when the
        check is not applicable (missing submission or submit script).
    """
    submission = Path(output_dir) / "submission.csv"
    if not submission.is_file():
        return None, "submission.csv not found"
    submit_script = _submit_script_path()
    if submit_script is None:
        return None, "submit.py not found"
    import os

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    pythonpath = [str(submit_script.parent)]
    if existing:
        pythonpath.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env["PYTHONIOENCODING"] = "utf-8"
    command = [
        sys.executable,
        str(submit_script),
        str(submission),
        "--check",
        "--data_dir",
        str(data_dir),
    ]
    try:
        # S603: intentional — official baseline submit.py check with
        # list-form args (no shell), fixed data_dir, and a timeout.
        completed = subprocess.run(  # noqa: S603
            command,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_SUBMIT_CHECK_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False, f"submit.py --check timed out after {_SUBMIT_CHECK_TIMEOUT}s"
    except OSError as exc:
        return False, f"submit.py --check failed to run: {exc}"
    if completed.returncode == 0:
        message = (completed.stdout or completed.stderr or "").strip()
        match = re.search(r"(?:校验通过[:：]|verified[:\s]*)\s*([\d,]+)\s*(?:行|rows)", message)
        if match:
            return True, f"format and alignment verified ({match.group(1)} rows)"
        return True, "format and alignment verified"
    detail = (completed.stderr or completed.stdout or "verification failed").strip()
    tail = next((ln for ln in reversed(detail.splitlines()) if ln.strip()), detail)
    return False, _ascii_safe(tail) or "verification failed"
