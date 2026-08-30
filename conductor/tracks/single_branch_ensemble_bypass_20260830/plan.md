# Implementation Plan: Single-Branch Ensembling Bypass

## Phase 1: Ensemble Pipeline Fast-Path (TDD) `[checkpoint: d47265d]`
- [x] Task: Write failing unit tests for single-candidate ensemble handling in `tests/ensembling/test_pipeline.py` (`d47265d`)
    - [x] Test `EnsemblePipeline.run` with a single candidate artifact returns immediately with 0 rounds executed and zero LLM calls
    - [x] Test `EnsemblePipeline.run` with `rounds=0` returns best individual candidate artifact directly
    - [x] Test `EnsemblePipeline.run` with empty solutions raises `ValueError`
    - [x] Verify tests fail as expected (Red phase)
- [x] Task: Implement single-solution and zero-rounds fast-path in `src/problem_2_v2/ensembling/pipeline.py` (`d47265d`)
    - [x] Add input validation and instant-return logic in `EnsemblePipeline.run`
    - [x] Ensure `EnsembleResult` returned has `rounds_executed=0`, `best_code`, and `optimal_solution` populated
    - [x] Verify all ensembling pipeline tests pass (Green phase)
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Master Orchestrator Integration & E2E Verification (TDD)
- [ ] Task: Write tests for single-branch ensembling bypass in `tests/test_orchestrator.py`
    - [ ] Test orchestrator with `num_branches=1` skips ensembling stage and passes candidate code directly to `finalizer.produce`
    - [ ] Test console announcements indicate that Stage 3 Ensembling is skipped
    - [ ] Test orchestrator with `ensemble_rounds=0` skips ensembling even when multiple branches exist
    - [ ] Verify tests fail as expected (Red phase)
- [ ] Task: Update Stage 3 bypass logic in `src/problem_2_v2/orchestrator.py`
    - [ ] Update Stage 3 guard: `if len(artifacts) > 1 and self.config.ensemble_rounds > 0:`
    - [ ] Add `elif len(artifacts) == 1:` console notification and direct pass-through
    - [ ] Add `elif len(artifacts) > 1 and self.config.ensemble_rounds == 0:` notification and direct pass-through
    - [ ] Verify orchestrator and full test suite pass cleanly (Green phase)
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)
