# Implementation Plan: Execution Environment & Guardrail Modules

## Phase 1: Unified Execution Guardrail Pipeline [checkpoint: c9c7034]
- [x] Task: Write failing tests for ExecutionConfig and ExecutionGuardrailPipeline
    - [x] Create `tests/execution/test_pipeline.py` testing sequential Leakage → Usage → Sandbox → Debugger orchestration
    - [x] Test `ExecutionConfig` controls (timeout, max_debug_rounds, guardrail toggles)
    - [x] Test graceful degradation when individual guardrail LLM calls fail
    - [x] Test successful execution returns validated `ExecutionResult` with parsed score
- [~] Task: Implement `ExecutionConfig` and `ExecutionGuardrailPipeline` in `src/problem_2_v2/execution/pipeline.py`
    - [x] Define `ExecutionConfig` Pydantic model with timeout, retry, and toggle settings
    - [x] Implement unified `run(script, task_spec) -> ExecutionResult` method
    - [x] Wire existing `DataLeakageCheckerAgent`, `DataUsageCheckerAgent`, `SubprocessRunner`, and `DebuggerAgent`
    - [x] Add Logfire span tracing for each pipeline stage
    - [x] Verify pipeline tests pass (c9c7034)
- [x] Task: Create `src/problem_2_v2/execution/__init__.py` with package exports
    - [x] Re-export `ExecutionGuardrailPipeline`, `ExecutionConfig`, and `FinalArtifactProducer` (FinalArtifactProducer export added in Phase 2, c9c7034)
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Final Artifact Producer ($\mathcal{A}_{\text{finalizer}}$) [checkpoint: ea0c7e8]
- [x] Task: Write failing tests for FinalArtifactProducer
    - [x] Create `tests/execution/test_finalizer.py` testing subsampling removal, model serialization injection, `metrics.json` export, and `./final/` output structure
    - [x] Test AST validation of rewritten production script
    - [x] Test extended timeout execution with DebuggerAgent fallback
- [~] Task: Implement `FinalArtifactProducer` in `src/problem_2_v2/execution/finalizer.py`
    - [x] Implement Pydantic AI finalizer agent prompt for subsampling removal and serialization
    - [x] Define `FinalArtifact` Pydantic model (script, output_dir, model_paths, metrics, submission_path)
    - [x] Integrate with `SubprocessRunner` (production timeout) and `DebuggerAgent`
    - [x] Verify finalizer tests pass (ea0c7e8)
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 3: Pipeline Integration Refactor [checkpoint: 88dfc0e]
- [x] Task: Write failing tests for refactored RefinementPipeline and EnsemblePipeline integration
    - [x] Create `tests/execution/test_integration.py` verifying both pipelines delegate to `ExecutionGuardrailPipeline`
    - [x] Verify all existing 252+ tests continue to pass (backward compatibility)
- [x] Task: Refactor `RefinementPipeline` to use `ExecutionGuardrailPipeline.run()`
    - [x] Replace direct guardrail and runner calls with unified pipeline delegation
    - [x] Verify refinement tests pass
- [x] Task: Refactor `EnsemblePipeline` to use `ExecutionGuardrailPipeline.run()`
    - [x] Replace direct guardrail and runner calls with unified pipeline delegation
    - [x] Verify ensemble tests pass
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 4: End-to-End Integration & Final Verification
- [x] Task: Write end-to-end integration test
    - [x] Create `tests/execution/test_e2e_execution.py` testing full guardrail → execution → finalization chain
    - [x] Test complete flow: candidate script → guardrails → sandbox → debugger → finalizer → `./final/` output (c623c61)
- [x] Task: Run full test suite and coverage verification
    - [x] Execute `uv run pytest --cov=src --cov-report=term-missing`
    - [x] Verify >80% coverage on new `execution/` modules (pipeline 97%, finalizer 93%, __init__ 100%)
    - [x] Verify all tests pass (273 passed, 94.64% total)
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)
