# Specification: Unified End-to-End Iteration Logging

## Overview
This track implements a unified, central iteration logging mechanism across all 4 stages of the MLE-STAR pipeline. Every candidate evaluation, model merge, ablation study, inner patch, ensemble round, and final production run is recorded as structured JSON records into `runs/<run_id>/iteration_logs.jsonl`. 

This strictly adheres to **Competition Requirement 5 (Run-Log Requirements)** and **Deliverable #3 (Run & Iteration Logs)** used by judges to evaluate **Autonomy** (Impact & Relevance) and **Robustness** (Technical Execution).

---

## Competition Run-Log Schema Specification

Every record in `runs/<run_id>/iteration_logs.jsonl` must contain:

| Field Name | Type | Description / Competition Requirement |
| :--- | :--- | :--- |
| `iteration_id` | `str` | Unique human-readable iteration tag (e.g. `cand_1`, `merge_1`, `b1_t0_k0`, `ens_r0`, `final_prod`) |
| `stage` | `str` | Pipeline stage: `INITIALIZATION`, `REFINEMENT`, `ENSEMBLING`, `FINALIZATION` |
| `hypothesis` | `str` | **Requirement 1:** What the agent intended to try and why (natural language rationale) |
| `code_diff` | `str` | **Requirement 2:** Unified diff representing the exact code modification applied |
| `metrics` | `dict[str, float]` | **Requirement 3:** Resulting metrics dictionary (e.g. `{"primary": 0.7381, "GAUC": 0.7612, "nDCG@5": 0.7150}`) |
| `validation_score` | `float | None` | Primary evaluation score |
| `delta_from_baseline` | `float | None` | Signed delta relative to the baseline anchor |
| `error_recovery_events` | `list[str]` | **Requirement 4:** Any syntax errors, exceptions, timeouts, guardrail repairs, or debugger recovery rounds |
| `success` | `bool` | Whether the iteration produced an executable, score-bearing artifact |
| `target_component` | `str | None` | Targeted module (e.g. `FEATURE_ENGINEERING`, `LOSS_AND_OPTIMIZER`) |
| `branch_index` | `int | None` | Parallel branch seed index |
| `timestamp` | `str` | ISO-8601 timestamp |
| `duration_seconds` | `float | None` | Execution runtime in seconds |

---

## Functional Requirements by Stage

1. **Stage 1 (Initialization & Model Merging):**
   - Record each retrieved candidate evaluation with its model description, hypothesis rationale, code diff against empty/starter, and validation score.
   - Record each greedy merge attempt with merge hypothesis, diff against previous merge, score, and acceptance status.

2. **Stage 2 (Refinement Loops):**
   - Record ablation studies with component variation hypotheses, score deltas, and identified bottlenecks.
   - Record every inner patch attempt with outer index ($t$), inner index ($k$), branch index, hypothesis plan, code diff, validation metrics, and any debugger retry errors.

3. **Stage 3 (Ensembling):**
   - Record each ensemble strategy proposal (Weighted Average, Stacking, Rank Average) with strategy hypothesis, combination code diff, validation metrics, and delta over best individual candidate.

4. **Stage 4 (Production Finalizer):**
   - Record final full-dataset training hypothesis, code diff against validation script, serialized model paths, metrics breakdown, and `submit.py --check` validation status.

---

## Non-Functional Requirements
- **Thread & Process Safety:** Thread-safe append writes to `runs/<run_id>/iteration_logs.jsonl` with unbuffered flushing.
- **Backward Compatibility:** Existing `plan` and `errors` property aliases preserved for zero breaking changes.
- **Testing & Quality:** >80% unit test coverage for contracts and pipeline hooks; passes `mypy src` and `ruff check src`.
