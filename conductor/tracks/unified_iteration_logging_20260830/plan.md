# Implementation Plan: Unified End-to-End Iteration Logging

## Phase 1: Unified Log Schema & Central Logger Hook (TDD)
- [ ] Task: Write failing unit tests for unified `IterationLogEntry` schema and central logger in `tests/test_iteration_logging.py`
- [ ] Task: Define unified `IterationLogEntry` contract and thread-safe central logger utility in `src/problem_2_v2/contracts/iteration.py`
- [ ] Task: Phase 1 Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Stage 1 (Initialization) & Stage 2 (Refinement) Integration (TDD)
- [ ] Task: Write failing unit tests for Stage 1 candidate/merge and Stage 2 ablation/refinement logging in `tests/test_initialization_pipeline.py` and `tests/test_refinement_pipeline.py`
- [ ] Task: Hook Stage 1 candidate evaluations and greedy merges to central `iteration_logs.jsonl` in `src/problem_2_v2/initialization/pipeline.py`
- [ ] Task: Hook Stage 2 ablation studies and inner refinement patches to central `iteration_logs.jsonl` in `src/problem_2_v2/refinement/pipeline.py`
- [ ] Task: Phase 2 Verification & Checkpoint (Refer to workflow.md)

## Phase 3: Stage 3 (Ensembling), Stage 4 (Finalizer) & E2E Validation (TDD)
- [ ] Task: Write failing unit tests for Stage 3 ensembling and Stage 4 finalizer logging in `tests/test_ensembling_pipeline.py` and `tests/test_finalizer.py`
- [ ] Task: Hook Stage 3 ensembling and Stage 4 production finalizer to central `iteration_logs.jsonl` in `src/problem_2_v2/ensembling/pipeline.py` and `src/problem_2_v2/execution/finalizer.py`
- [ ] Task: Run full test suite and verify end-to-end execution logging
- [ ] Task: Phase 3 Verification & Checkpoint (Refer to workflow.md)
