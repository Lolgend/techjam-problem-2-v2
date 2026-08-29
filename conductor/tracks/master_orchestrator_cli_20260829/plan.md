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

## Phase 2: Master Configuration & Orchestrator (MLEStarPipeline)
- [ ] Task: Write failing tests for MLEStarConfig, MLEStarPipeline, and MLEStarResult
    - [ ] Create `tests/test_orchestrator.py` testing configuration defaults, 5-stage coordination, baseline delta calculation, and dry-run validation
- [ ] Task: Implement `MLEStarConfig` in `src/problem_2_v2/config.py`
    - [ ] Define hyperparameter configuration with validation and serialization
- [ ] Task: Implement `MLEStarPipeline` and `MLEStarResult` in `src/problem_2_v2/orchestrator.py`
    - [ ] Wire Ingestion -> Parallel Branches -> Ensembling -> Final Artifact Production -> Baseline Delta
    - [ ] Implement sync `run()` and async `run_async()`
    - [ ] Verify orchestrator tests pass
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 3: Command-Line Interface (cli.py & __init__.py:main)
- [ ] Task: Write failing tests for CLI commands and argument parser
    - [ ] Create `tests/test_cli.py` testing `run` command flags, `--dry-run`, `--version`, and invalid argument handling
- [ ] Task: Implement CLI entry point in `src/problem_2_v2/cli.py`
    - [ ] Implement argparse parser with subcommands `run` and `version`
    - [ ] Update `src/problem_2_v2/__init__.py` to export `main`, `MLEStarPipeline`, `MLEStarConfig`, and `MLEStarResult`
    - [ ] Verify CLI tests pass
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 4: Full End-to-End System Integration & Final Verification
- [ ] Task: Write end-to-end master pipeline integration test
    - [ ] Create `tests/test_e2e_master.py` testing complete execution from raw markdown task to final `./final/` production artifacts
- [ ] Task: Run full test suite and verify 100% pass rate and coverage
    - [ ] Execute `uv run pytest --cov=src --cov-report=term-missing`
    - [ ] Verify all tests pass
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)
