# Implementation Plan: Unified Run-Level Iteration Logging

## Phase 1: Contract & Logger Refactoring (TDD)
- [x] Task: Write failing unit tests for `root_run_id` and single-file path resolution in `tests/test_iteration_logging.py` (`74d3549`)
    - [x] Add test cases for `root_run_id` across POSIX and Windows hierarchical run IDs
    - [x] Add test cases verifying `for_run` resolves nested branch and final namespaces to the root run log file
    - [x] Add test cases for path-keyed instance synchronization and multi-threaded append integrity
    - [x] Verify tests fail as expected (Red phase)
- [x] Task: Implement `root_run_id` and path-synchronized registry in `src/problem_2_v2/contracts/iteration.py` (`74d3549`)
    - [x] Define `root_run_id(run_id: str) -> str`
    - [x] Update `CentralIterationLogger.for_run` to resolve to the root run path
    - [x] Implement path-keyed instance caching and thread locking
    - [x] Verify all iteration logging unit tests pass (Green phase)
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Pipeline Integration & End-to-End Verification
- [ ] Task: Update and expand end-to-end master logging tests in `tests/test_e2e_master.py`
    - [ ] Update `test_master_run_streams_unified_iteration_logs` to assert strictly one root `iteration_logs.jsonl`
    - [ ] Assert absence of nested `branch_0/iteration_logs.jsonl` and `final/iteration_logs.jsonl` files
    - [ ] Verify all 4 stage entries (`INITIALIZATION`, `REFINEMENT`, `ENSEMBLING`, `FINALIZATION`) are present in sequential append order
    - [ ] Fix default candidates assertion in `tests/test_orchestrator.py`
- [ ] Task: Verify pipeline stages and complete test suite
    - [ ] Run full test suite (`uv run pytest`) and verify 100% pass rate
    - [ ] Run type check (`uv run mypy src tests`) and linter (`uv run ruff check src tests`)
    - [ ] Check code coverage (`uv run pytest --cov=src --cov-report=term-missing`)
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)
