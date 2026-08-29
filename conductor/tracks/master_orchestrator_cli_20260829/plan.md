# Implementation Plan: Master Orchestrator, CLI Interface, and Package API Unification

## Phase 1: Package API Unification (__init__.py Exports) [checkpoint: e2f1c38]
- [x] Task: Write failing tests for submodule imports and __all__ exports
    - [x] Create `tests/test_package_exports.py` testing imports and `__all__` completeness across all 7 submodules
- [x] Task: Implement `__init__.py` files for all subpackages
    - [x] Add `src/problem_2_v2/ingestion/__init__.py`
    - [x] Add `src/problem_2_v2/search/__init__.py`
    - [x] Add `src/problem_2_v2/initialization/__init__.py`
    - [x] Add `src/problem_2_v2/refinement/__init__.py`
    - [x] Add `src/problem_2_v2/guardrails/__init__.py`
    - [x] Add `src/problem_2_v2/runner/__init__.py`
    - [x] Add `src/problem_2_v2/ensembling/__init__.py`
    - [x] Verify export tests pass (e2f1c38)
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Master Configuration & Orchestrator (MLEStarPipeline) [checkpoint: cd13a02]
- [x] Task: Write failing tests for MLEStarConfig, MLEStarPipeline, and MLEStarResult
    - [x] Create `tests/test_orchestrator.py` testing configuration defaults, 5-stage coordination, baseline delta calculation, and dry-run validation
- [x] Task: Implement `MLEStarConfig` in `src/problem_2_v2/config.py`
    - [x] Define hyperparameter configuration with validation and serialization
- [x] Task: Implement `MLEStarPipeline` and `MLEStarResult` in `src/problem_2_v2/orchestrator.py`
    - [x] Wire Ingestion -> Parallel Branches -> Ensembling -> Final Artifact Production -> Baseline Delta
    - [x] Implement sync `run()` and async `run_async()`
    - [x] Verify orchestrator tests pass (cd13a02)
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 3: Command-Line Interface (cli.py & __init__.py:main) [checkpoint: 2533bd0]
- [x] Task: Write failing tests for CLI commands and argument parser
    - [x] Create `tests/test_cli.py` testing `run` command flags, `--dry-run`, `--version`, and invalid argument handling
- [x] Task: Implement CLI entry point in `src/problem_2_v2/cli.py`
    - [x] Implement argparse parser with subcommands `run` and `version`
    - [x] Update `src/problem_2_v2/__init__.py` to export `main`, `MLEStarPipeline`, `MLEStarConfig`, and `MLEStarResult`
    - [x] Verify CLI tests pass (2533bd0)
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 4: Full End-to-End System Integration & Final Verification [checkpoint: f1e1344]
- [x] Task: Write end-to-end master pipeline integration test
    - [x] Create `tests/test_e2e_master.py` testing complete execution from raw markdown task to final `./final/` production artifacts
- [x] Task: Run full test suite and verify 100% pass rate and coverage
    - [x] Execute `uv run pytest --cov=src --cov-report=term-missing`
    - [x] Verify all tests pass (307 passed, 95.32%; orchestrator 100%, cli 91%, config 100%)
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)
