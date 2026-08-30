# Specification: Unified Run-Level Iteration Logging

## Overview
This track refactors the MLE-STAR iteration logging infrastructure to unify all pipeline run records into a single, consolidated log file per run: `runs/<run_id>/iteration_logs.jsonl`. 

Previously, parallel seed branches and the final production stage wrote records into isolated sub-logs (`runs/<run_id>/branch_0/iteration_logs.jsonl` and `runs/<run_id>/final/iteration_logs.jsonl`), scattering run telemetry across directories. This track ensures that every pipeline stage—Stage 1 Initialization across all branches, Stage 2 Refinement across all branches, Stage 3 Adaptive Ensembling, and Stage 4 Production Finalization—streams structured `IterationLogEntry` records directly into the top-level run log file in real time.

---

## Functional Requirements

1. **Root Run Resolution:**
   - Implement `root_run_id(run_id: str) -> str` in `problem_2_v2.contracts.iteration` to strip sub-stage suffixes (such as `/branch_0`, `/branch_1`, `/final`, or Windows path separators) from nested run identifiers.
   - Update `CentralIterationLogger.for_run(runs_dir, run_id)` so it resolves the target file path to `Path(runs_dir) / root_run_id(run_id) / "iteration_logs.jsonl"`.

2. **Single-File Run Topology:**
   - Ensure that `runs/<run_id>/iteration_logs.jsonl` is the sole log file produced for any given run.
   - Stage directories (e.g. `runs/<run_id>/branch_0/`, `runs/<run_id>/final/`) retain only their execution sandboxes (e.g. `sandbox_cand1`, `sandbox_final`) with zero nested `iteration_logs.jsonl` files.

3. **Continuous Real-Time Streaming & Thread Safety:**
   - Maintain a path-synchronized instance registry in `CentralIterationLogger` with thread locking (`threading.Lock`), ensuring that concurrent parallel branches write atomic, uncorrupted JSON lines with immediate line-flushing (`buffering=1`, `handle.flush()`).
   - Preserve all entry metadata, including `branch_index` (0, 1, ... for parallel branch iterations; `None` for ensembling and finalizer iterations), `timestamp`, `stage`, `hypothesis`, `code_diff`, `metrics`, `validation_score`, `delta_from_baseline`, `error_recovery_events`, and `duration_seconds`.

4. **Backward Compatibility & Pipeline Integration:**
   - Preserve all existing method signatures across `InitializationPipeline`, `RefinementPipeline`, `EnsemblePipeline`, and `FinalArtifactProducer`.
   - Keep existing aliases (`plan` -> `hypothesis`, `errors` -> `error_recovery_events`) intact on `IterationLogEntry`.

---

## Non-Functional Requirements
- **Thread & Process Safety:** Concurrent writes across thread pools in `ParallelSolutionGenerator` are serialized safely without file-locking deadlocks or race conditions.
- **Coverage & Quality Gates:** All unit and integration tests pass with 100% green status; passes `mypy src tests` and `ruff check src tests`.
- **Zero Breaking Changes:** Maintains full compatibility with existing result models and CLI output workflows.

---

## Acceptance Criteria
- [ ] `CentralIterationLogger.for_run(runs_dir, "run_id/branch_0")` and `CentralIterationLogger.for_run(runs_dir, "run_id/final")` return a logger pointing to `runs/run_id/iteration_logs.jsonl`.
- [ ] Concurrent writes from multiple threads appending to the same run logger produce valid JSON lines with no missing or corrupted records.
- [ ] Running an end-to-end multi-branch pipeline creates exactly one `iteration_logs.jsonl` under `runs/<run_id>/` and no log files inside branch or final subdirectories.
- [ ] The unified log contains entries for `INITIALIZATION`, `REFINEMENT`, `ENSEMBLING`, and `FINALIZATION` stages.
