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
from problem_2_v2.orchestrator import MLEStarPipeline

_PROVIDERS = ("duckduckgo", "tavily", "google", "mock")


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
    config = MLEStarConfig(
        model=args.model,
        search_provider=args.search_provider,
        num_branches=args.branches,
        outer_loops=args.outer_loops,
        inner_loops=args.inner_loops,
        ensemble_rounds=args.ensemble_rounds,
        seeds=_parse_seeds(args.seeds),
    )
    pipeline = MLEStarPipeline(config=config)

    if args.dry_run:
        return _dry_run(pipeline, args)

    try:
        result = pipeline.run(args.task, args.data)
    except (FileNotFoundError, NotADirectoryError) as exc:
        parser.error(str(exc))
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1

    print(f"Run completed in {result.duration_seconds:.1f}s")
    print(
        f"Baseline: {result.baseline_score:.4f}  "
        f"Final: {result.final_score:.4f}  Delta: {result.score_delta:.4f}"
    )
    if result.final_artifact is not None:
        _copy_final_output(result.final_artifact.output_dir, args.output)
        print(f"Artifacts written to {args.output}")
    return 0 if result.success else 1


def _dry_run(pipeline: MLEStarPipeline, args: argparse.Namespace) -> int:
    """Validate task and dataset parsing without executing the pipeline."""
    try:
        spec = pipeline.validate(args.task, args.data)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Dry-run OK: task '{spec.task_name}', baseline {spec.baseline_score:.4f}")
    return 0


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
        if item.is_file():
            shutil.copy2(item, dst / item.name)
