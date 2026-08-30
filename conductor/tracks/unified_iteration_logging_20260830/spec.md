# Specification: Unified End-to-End Iteration Logging

## Overview
This track implements a unified, central iteration logging mechanism across all 4 stages of the MLE-STAR pipeline. Instead of fragmenting logs or only recording Stage 3, every candidate evaluation, model merge, ablation study, inner patch, ensemble round, and final production artifact will be recorded as structured JSON records into `runs/<run_id>/iteration_logs.jsonl`. This fulfills 100% of the competition's Deliverable #3 (Run & Iteration Logs) requirements.

## Functional Requirements
1. **Unified File Target (`runs/<run_id>/iteration_logs.jsonl`):**
   - Stream all lifecycle events directly to a single root log file in real time using append mode (`"a"`).

2. **Stage 1 (Initialization & Model Merging) Logging:**
   - Log each retrieved model card evaluation with model name, rationale, code diff, validation score, and delta.
   - Log each greedy merge step with merge hypothesis, diff, validation score, delta, and acceptance status.

3. **Stage 2 (Refinement Loops) Logging:**
   - Log ablation studies with tested component variations and impact scores.
   - Log each inner refinement patch with outer iteration ($t$), inner iteration ($k$), branch index, hypothesis plan, code diff, validation score, delta, and error recovery events.

4. **Stage 3 (Ensembling) Logging:**
   - Log each ensemble strategy proposal (Weighted Average, Stacking, Rank Average) with plan, code diff, validation score, and delta over best individual candidate.

5. **Stage 4 (Production Finalizer) Logging:**
   - Log final production training on the complete dataset, serialized model locations, final validation metrics, and submission output verification.

## Non-Functional Requirements
- **Resilience:** Thread-safe / process-safe logging with immediate flush.
- **Coverage:** Maintain >80% unit test coverage for logging contracts and pipeline hooks.
- **Typing & Linting:** Pass `mypy src` and `ruff check src`.

## Acceptance Criteria
- [ ] Central `iteration_logs.jsonl` is created at the start of `mlestar run`.
- [ ] Every candidate, merge, ablation, refinement, ensemble, and finalization step appends a valid JSON record.
- [ ] Tests verify that all stage logs conform to the schema.
