"""Command-line interface for the ``problem-2-v2`` package.

Provides the ``run`` subcommand (full pipeline or ``--dry-run``) and the
``version`` subcommand, wired to the package console script.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from importlib import metadata
from pathlib import Path

from problem_2_v2.config import MLEStarConfig
from problem_2_v2.console import announce, format_delta, format_score
from problem_2_v2.contracts.task import TaskSpecification
from problem_2_v2.orchestrator import MLEStarPipeline, MLEStarResult, configure_event_loop_policy

_PROVIDERS = ("duckduckgo", "tavily", "google", "mock")
_BANNER_WIDTH = 78


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
        "--dry-run", action="store_true", help="Validate inputs without running the pipeline."
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
    _print_summary(result, args.output)
    return 0 if result.success else 1


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


def _print_summary(result: MLEStarResult, output_dir: str) -> None:
    """Render the final summary box with duration, scores, and artifacts."""
    announce("=" * _BANNER_WIDTH)
    announce(f"Run complete in {result.duration_seconds:.1f}s")
    announce(
        f"Baseline: {result.baseline_score:.4f} | "
        f"Final: {format_score(result.final_score)} | Delta: {format_delta(result.score_delta)}"
    )
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
