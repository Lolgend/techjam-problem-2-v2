# Specification: Execution Environment & Guardrail Modules

## 1. Overview
This track unifies the existing execution safety components (SubprocessRunner, DebuggerAgent, DataLeakageCheckerAgent, DataUsageCheckerAgent) into a single reusable `ExecutionGuardrailPipeline` orchestrator, and introduces the `FinalArtifactProducer` agent ($\mathcal{A}_{\text{finalizer}}$) that strips subsampling constraints from the winning solution, ensures full-dataset training, and serializes models and evaluation metrics to disk. This completes the end-to-end MLE-STAR pipeline from problem ingestion to production-ready artifact output.

## 2. Functional Requirements

### A. Unified Execution Guardrail Pipeline (`src/problem_2_v2/execution/pipeline.py`)
- `ExecutionGuardrailPipeline`:
  - Accepts a candidate Python script string and `TaskSpecification`.
  - Orchestrates the following sequential pass:
    1. **Data Leakage Check** ($\mathcal{A}_{\text{leakage}}$): Inspects preprocessing blocks for train/validation data leakage. If detected, auto-repairs and replaces the offending code block.
    2. **Data Usage Check** ($\mathcal{A}_{\text{data}}$): Verifies all multimodal data sources from `TaskSpecification` are actively loaded. If omissions found, augments data loading section.
    3. **Subprocess Sandbox Execution**: Runs the (possibly corrected) script in an isolated subprocess with timeout, memory limits, and `./input` directory mapping.
    4. **Automatic Debugger Loop** ($\mathcal{A}_{\text{debugger}}$): If execution fails (non-zero exit, unhandled exception), captures traceback and retries up to `max_debug_rounds` (default 3) before rolling back.
  - Returns a validated `ExecutionResult` with success flag, stdout/stderr, parsed validation score, and runtime duration.
  - Exposes configuration via `ExecutionConfig` dataclass (timeout, max_debug_rounds, enable_leakage_check, enable_usage_check).

### B. Final Artifact Producer (`src/problem_2_v2/execution/finalizer.py`)
- `FinalArtifactProducer` ($\mathcal{A}_{\text{finalizer}}$):
  - Accepts the winning solution script $s^*_{\text{ens}}$ and `TaskSpecification`.
  - Prompts a Pydantic AI agent to:
    1. Identify and remove all subsampling/row-capping constraints (e.g., `.head(30000)`, `.sample(n=30000)`, `[:30000]`).
    2. Ensure the script trains on the **complete** dataset.
    3. Add model serialization (`joblib.dump`, `torch.save`, or framework-appropriate method).
    4. Add JSON metrics export (`metrics.json`) with final evaluation scores.
    5. Preserve `./final/submission.csv` output.
  - Validates the rewritten script via AST syntax check.
  - Executes the production script in `SubprocessRunner` (with extended timeout for full-data training) and `DebuggerAgent` fallback.
  - Returns a `FinalArtifact` model containing: final script code, output directory path (`./final/`), model file paths, metrics dict, and submission CSV path.

### C. Execution Configuration & Shared Module Exports (`src/problem_2_v2/execution/__init__.py`)
- `ExecutionConfig`: Pydantic model for pipeline configuration (timeout_seconds, max_debug_rounds, sandbox_base_dir, enable_leakage_check, enable_usage_check, production_timeout_seconds).
- Re-exports `ExecutionGuardrailPipeline`, `FinalArtifactProducer`, `ExecutionConfig` from the `execution` package.

### D. Integration with Existing Pipelines
- Refactor `RefinementPipeline` and `EnsemblePipeline` to delegate script execution to `ExecutionGuardrailPipeline.run()` instead of directly calling individual guardrails and runners.
- Ensure backward compatibility: all existing tests continue to pass.

## 3. Non-Functional Requirements
- **Robustness:** Graceful degradation when guardrail checks encounter LLM failures (skip check with warning, proceed to execution).
- **Observability:** Logfire span tracing for each guardrail stage (leakage_check, usage_check, sandbox_exec, debug_retry, finalization).
- **Coverage:** >80% unit and integration test coverage across all new modules.

## 4. Acceptance Criteria
- [ ] `ExecutionGuardrailPipeline` orchestrates Leakage → Usage → Sandbox → Debugger in a single `run()` call.
- [ ] `ExecutionConfig` controls pipeline behavior (timeouts, retry counts, guardrail toggles).
- [ ] `FinalArtifactProducer` strips subsampling, adds model serialization, and produces `./final/` output.
- [ ] Production script executes successfully on full dataset with extended timeout.
- [ ] `RefinementPipeline` and `EnsemblePipeline` use `ExecutionGuardrailPipeline` instead of ad-hoc calls.
- [ ] All existing 252 tests continue to pass (backward compatibility).
- [ ] New unit and integration tests pass with >80% coverage on new modules.

## 5. Out of Scope
- GPU resource accounting and quota management (future track).
- Container/Docker-based sandboxing (subprocess isolation is sufficient for now).
- Model registry or deployment pipeline integration.
