# Specification: Single-Branch Ensembling Bypass

## Overview
This track introduces a conditional bypass for Stage 3 (Adaptive Ensembling) when running the MLE-STAR pipeline with a single branch ($L=1$), when only a single candidate solution artifact is produced, or when `ensemble_rounds == 0`.

Currently, `orchestrator.py` checks `if artifacts:` before triggering Stage 3 ensembling. For $L=1$, this runs 3 rounds of ensembling on a single solution artifact against itself, creating redundant LLM API calls and execution latency. This track ensures that single-candidate runs automatically skip Stage 3 with an informational console announcement and directly pass the candidate solution to Stage 4 (Production Finalizer). In addition, `EnsemblePipeline.run` is updated with a defensive fast-path return when invoked with $\le 1$ solutions.

---

## Functional Requirements

1. **Orchestrator Stage 3 Conditional Bypass:**
   - In `MLEStarPipeline.run_async`, trigger Stage 3 Adaptive Ensembling only when `len(artifacts) > 1 and self.config.ensemble_rounds > 0`.
   - If `len(artifacts) == 1`:
     - Announce: `"[Stage 3/4] Adaptive Ensembling skipped (single candidate; forwarding to finalizer)..."`.
     - Set `ensemble_result = None` and forward `artifacts[0].full_code` directly to Stage 4 Finalizer.
   - If `len(artifacts) > 1 and self.config.ensemble_rounds == 0`:
     - Announce: `"[Stage 3/4] Adaptive Ensembling skipped (ensemble_rounds=0; selecting best candidate)..."`.
     - Forward the highest-scoring candidate artifact directly to Stage 4 Finalizer.
   - If `len(artifacts) == 0`:
     - Issue a warning that no candidates were produced across branches and skip Stage 3 and Stage 4.

2. **Ensemble Pipeline Fast-Path & Input Validation:**
   - In `EnsemblePipeline.run(spec, solutions, run_id)`:
     - If `len(solutions) == 0`: raise `ValueError("No candidate solutions provided for ensembling.")`.
     - If `len(solutions) == 1` or `self.rounds <= 0`: immediately return `EnsembleResult` holding the single / best candidate solution with `rounds_executed=0`, `attempts=[]`, without invoking `EnsemblePlannerAgent` or `EnsemblerAgent`.

---

## Non-Functional Requirements
- **Efficiency & Resource Conservation:** Zero LLM agent invocations and zero extra sandbox executions during Stage 3 for single-candidate runs.
- **Coverage & Quality Gates:** Passes `uv run pytest`, `uv run mypy src tests`, and `uv run ruff check src tests` with >80% coverage on new code.
- **Zero Breaking Changes:** `MLEStarResult` structure and CLI return codes remain fully backward-compatible.

---

## Acceptance Criteria
- [ ] Running `MLEStarPipeline.run_async` with `num_branches=1` skips Stage 3 ensembling and proceeds directly to Stage 4.
- [ ] Running `MLEStarPipeline.run_async` with `ensemble_rounds=0` skips Stage 3 ensembling and uses the top-performing candidate.
- [ ] `EnsemblePipeline.run` with 1 artifact returns immediately with `rounds_executed=0` and 0 LLM calls.
- [ ] All unit and integration tests pass with 100% green status.
