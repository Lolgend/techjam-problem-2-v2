# Implementation Plan: Execution Environment & Guardrail Modules

## Phase 1: Unified Execution Guardrail Pipeline
- [ ] Task: Write failing tests for ExecutionConfig and ExecutionGuardrailPipeline
    - [ ] Create `tests/execution/test_pipeline.py` testing sequential Leakage → Usage → Sandbox → Debugger orchestration
    - [ ] Test `ExecutionConfig` controls (timeout, max_debug_rounds, guardrail toggles)
    - [ ] Test graceful degradation when individual guardrail LLM calls fail
    - [ ] Test successful execution returns validated `ExecutionResult` with parsed score
- [ ] Task: Implement `ExecutionConfig` and `ExecutionGuardrailPipeline` in `src/problem_2_v2/execution/pipeline.py`
    - [ ] Define `ExecutionConfig` Pydantic model with timeout, retry, and toggle settings
    - [ ] Implement unified `run(script, task_spec) -> ExecutionResult` method
    - [ ] Wire existing `DataLeakageCheckerAgent`, `DataUsageCheckerAgent`, `SubprocessRunner`, and `DebuggerAgent`
    - [ ] Add Logfire span tracing for each pipeline stage
    - [ ] Verify pipeline tests pass
- [ ] Task: Create `src/problem_2_v2/execution/__init__.py` with package exports
    - [ ] Re-export `ExecutionGuardrailPipeline`, `ExecutionConfig`, and `FinalArtifactProducer`
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Final Artifact Producer ($\mathcal{A}_{\text{finalizer}}$)
- [ ] Task: Write failing tests for FinalArtifactProducer
    - [ ] Create `tests/execution/test_finalizer.py` testing subsampling removal, model serialization injection, `metrics.json` export, and `./final/` output structure
    - [ ] Test AST validation of rewritten production script
    - [ ] Test extended timeout execution with DebuggerAgent fallback
- [ ] Task: Implement `FinalArtifactProducer` in `src/problem_2_v2/execution/finalizer.py`
    - [ ] Implement Pydantic AI finalizer agent prompt for subsampling removal and serialization
    - [ ] Define `FinalArtifact` Pydantic model (script, output_dir, model_paths, metrics, submission_path)
    - [ ] Integrate with `SubprocessRunner` (production timeout) and `DebuggerAgent`
    - [ ] Verify finalizer tests pass
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 3: Pipeline Integration Refactor
- [ ] Task: Write failing tests for refactored RefinementPipeline and EnsemblePipeline integration
    - [ ] Create `tests/execution/test_integration.py` verifying both pipelines delegate to `ExecutionGuardrailPipeline`
    - [ ] Verify all existing 252+ tests continue to pass (backward compatibility)
- [ ] Task: Refactor `RefinementPipeline` to use `ExecutionGuardrailPipeline.run()`
    - [ ] Replace direct guardrail and runner calls with unified pipeline delegation
    - [ ] Verify refinement tests pass
- [ ] Task: Refactor `EnsemblePipeline` to use `ExecutionGuardrailPipeline.run()`
    - [ ] Replace direct guardrail and runner calls with unified pipeline delegation
    - [ ] Verify ensemble tests pass
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 4: End-to-End Integration & Final Verification
- [ ] Task: Write end-to-end integration test
    - [ ] Create `tests/execution/test_e2e_execution.py` testing full guardrail → execution → finalization chain
    - [ ] Test complete flow: candidate script → guardrails → sandbox → debugger → finalizer → `./final/` output
- [ ] Task: Run full test suite and coverage verification
    - [ ] Execute `uv run pytest --cov=src --cov-report=term-missing`
    - [ ] Verify >80% coverage on new `execution/` modules
    - [ ] Verify all tests pass
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)
