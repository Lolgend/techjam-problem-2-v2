# Implementation Plan: Ablation Debugger Self-Repair Integration

## Phase 1: Test Suite & Failure Reproduction (TDD Red Phase) [checkpoint: 7d2974b]
- [x] Task: Write failing unit tests for ablation debugger integration [7d2974b]
  - [x] Add unit test in `tests/refinement/test_ablation.py` verifying `AblationSummarizerAgent` calls the debugger on syntax/runtime execution failure.
  - [x] Add unit test verifying that a repaired ablation script produces a valid `AblationReport`.
  - [x] Add unit test verifying graceful fallback when all debugger rounds fail.
  - [x] Run `uv run pytest tests/refinement/test_ablation.py` and confirm tests fail as expected (Red Phase).
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md) [7d2974b]

## Phase 2: Implementation of Ablation Debugger Integration (TDD Green Phase) [checkpoint: 25db1e7]
- [x] Task: Update `AblationSummarizerAgent` with Debugger Integration [e765b49]
  - [x] Update `AblationSummarizerAgent.__init__` to accept `debugger: DebuggerAgent | None = None` (with runner/model default fallback).
  - [x] Update `AblationSummarizerAgent.summarize()` to detect execution failure and trigger debugger self-repair across `max_debug_rounds`.
  - [x] Instrument with Logfire spans (`ablation.debug_repair`) and error logging.
- [x] Task: Update Refinement Pipeline and Orchestrator Wiring [25db1e7]
  - [x] Update `RefinementPipeline` to inject the branch debugger into `AblationSummarizerAgent`.
  - [x] Update `MLEStarPipeline._build_branch` in `src/problem_2_v2/orchestrator.py` to pass the branch debugger instance.
- [x] Task: Verify unit tests pass (Green Phase) [25db1e7]
  - [x] Run `uv run pytest tests/refinement/` to confirm all ablation tests pass.
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md) [25db1e7]

## Phase 3: End-to-End Verification & Quality Gates [checkpoint: 99210ba]
- [x] Task: Quality Gates & Full Verification [99210ba]
  - [x] Run `uv run ruff check src` and `uv run ruff format --check src`.
  - [x] Run `uv run mypy src`.
  - [x] Run `uv run pytest --cov=src --cov-report=term-missing` and verify >80% coverage on modified modules.
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md) [99210ba]
