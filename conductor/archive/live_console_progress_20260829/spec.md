# Specification: Live Real-Time Console Progress, Startup Banner & Stage Telemetry Streaming

## 1. Overview
This track implements real-time interactive terminal progress output, unbuffered stdout streaming (`flush=True`), an immediate startup banner, and stage announcements across Initialization, Targeted Refinement, Ensembling, and Finalization. This ensures users see live interactive progress, candidate evaluation scores, and delta updates at every step rather than waiting on a blank terminal.

## 2. Functional Requirements

### A. CLI Startup Banner & Summary Box (`src/problem_2_v2/cli.py`)
- **Startup Banner:** Printed immediately upon `problem-2-v2 run`:
  - Task name, task type, dataset directory, metric name, baseline score.
  - Active model identifier, search provider, and parallel branches/loops configuration.
- **Final Summary Box:** Printed upon run completion:
  - Total elapsed wall-clock duration.
  - Baseline score, Final validation score, and signed score delta ($\Delta$).
  - List of generated artifact files in `./final/` (submission.csv, metrics.json, model checkpoints).

### B. Master Orchestrator Stage Telemetry (`src/problem_2_v2/orchestrator.py`)
- Print clear stage boundaries with `flush=True`:
  - `[Stage 1/4] Launching L Parallel Seed Branches...`
  - `[Stage 2/4] Aggregating Candidate Artifacts...`
  - `[Stage 3/4] Adaptive Ensembling (R rounds)...`
  - `[Stage 4/4] Production Finalization (Full Dataset Training & Model Serialization)...`

### C. Sub-Pipeline Live Telemetry
- **Parallel Generator (`ensembling/parallel.py`):**
  - Live announcements of branch launch: `[Branch {index} (seed={seed})] Starting pipeline...`
  - Live announcements of branch completion: `[Branch {index} (seed={seed})] Finished with Score: {score:.4f}`
- **Initialization Pipeline (`initialization/pipeline.py`):**
  - Web search status: `[Search] Retrieving candidates via {provider}...`
  - Candidate evaluations: `[Candidate {i}/{M}] {name} -> Validation Score: {score:.4f}`
  - Model merger: `[Merge] Sequential merging completed. Initial s0 Score: {score:.4f}`
- **Targeted Refinement Pipeline (`refinement/pipeline.py`):**
  - Outer loop: `[Outer {t+1}/{T}] Running ablation study across components...`
  - Block extraction: `[Outer {t+1}/{T}] Extracted high-impact block: '{category}'`
  - Inner loop: `[Inner {t+1}.{k+1}/{K}] Plan: '{plan}' -> Score: {score:.4f} (Δ {delta:+.4f})`
- **Adaptive Ensembling Pipeline (`ensembling/pipeline.py`):**
  - Round status: `[Ensemble Round {r+1}/{R}] Strategy: '{method}' -> Score: {score:.4f} (Δ {delta:+.4f})`
- **Finalizer (`execution/finalizer.py`):**
  - Production status: `[Finalizer] Stripping subsampling and training on complete dataset...`
  - Completion: `[Finalizer] Production run complete. Score: {score:.4f}`

## 3. Non-Functional Requirements
- **Immediate Streaming:** All prints use `flush=True` so terminal emulators and shells render output with zero delay.
- **Robustness:** Formatting handles `None` scores, exceptions, and empty diffs gracefully.
- **Backward Compatibility:** All existing 310 tests continue to pass 100% green.

## 4. Acceptance Criteria
- [ ] Running `problem-2-v2 run` immediately renders the startup banner.
- [ ] Intermediate candidate scores, ablation summaries, and refinement plans stream live to the terminal.
- [ ] Final summary box accurately reports baseline, final score, delta, duration, and artifact paths.
- [ ] Full test suite passes 100% green.
