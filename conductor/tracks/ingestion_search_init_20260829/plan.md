# Implementation Plan: Ingestion & Search-Guided Initialization Phase

## Phase 1: Task Ingestion & Search Provider Layer
- [ ] Task: Write failing tests for TaskExtractor and SearchProviders
    - [ ] Create `tests/ingestion/test_extractor.py` testing markdown problem description extraction to `TaskSpecification`
    - [ ] Create `tests/search/test_providers.py` testing `MockSearchProvider`, `TavilySearchProvider`, and `DuckDuckGoSearchProvider`
- [ ] Task: Implement `TaskExtractor` agent in `src/problem_2_v2/ingestion/extractor.py`
    - [ ] Implement Pydantic AI agent with structured output bound to `TaskSpecification`
    - [ ] Implement fallback heuristic markdown extractor for offline/mock environments
    - [ ] Verify ingestion tests pass
- [ ] Task: Implement `SearchProvider` interfaces in `src/problem_2_v2/search/providers.py`
    - [ ] Define `SearchProvider` protocol and `SearchResult` model
    - [ ] Implement `MockSearchProvider`, `TavilySearchProvider`, `GoogleSearchProvider`, and `DuckDuckGoSearchProvider`
    - [ ] Verify provider tests pass
- [ ] Task: Write failing tests for `RetrieverAgent`
    - [ ] Create `tests/search/test_retriever.py` testing query construction, model card parsing, and candidate count
- [ ] Task: Implement `RetrieverAgent` in `src/problem_2_v2/search/retriever.py`
    - [ ] Implement query generation from `TaskSpecification`
    - [ ] Implement Pydantic AI retriever agent with Figure 9 prompt to output `RetrievedCandidates`
    - [ ] Verify retriever tests pass
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Execution Sandbox & Autonomous Debugger
- [ ] Task: Write failing tests for SubprocessRunner
    - [ ] Create `tests/runner/test_sandbox.py`
    - [ ] Test isolated sandbox directory creation and input data symlinking/copying
    - [ ] Test execution timeout, exit code capture, and metric parsing (`ExecutionResult`)
    - [ ] Test syntax error and exception capture
- [ ] Task: Implement `SubprocessRunner` in `src/problem_2_v2/runner/sandbox.py`
    - [ ] Implement directory workspace management and `./input` mapping
    - [ ] Implement subprocess execution with timeout and process termination
    - [ ] Implement stdout/stderr capture and validation score regex extraction
    - [ ] Verify sandbox tests pass
- [ ] Task: Write failing tests for DebuggerAgent
    - [ ] Create `tests/runner/test_debugger.py` testing automated traceback repair for syntax/import/runtime errors
- [ ] Task: Implement `DebuggerAgent` in `src/problem_2_v2/runner/debugger.py`
    - [ ] Implement Pydantic AI debugger agent with Figure 19 prompt
    - [ ] Implement iterative repair loop up to `max_debug_rounds`
    - [ ] Verify debugger tests pass
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 3: Candidate Code Generation & Evaluation ($\mathcal{A}_{\text{init}}$)
- [ ] Task: Write failing tests for CandidateEvaluatorAgent
    - [ ] Create `tests/initialization/test_evaluator.py`
    - [ ] Test candidate script generation prompt with `TaskSpecification` + `ModelCard`
    - [ ] Test hold-out validation and 30,000 row subsampling constraint enforcement
    - [ ] Test evaluation and score sorting ($\pi$ permutation)
- [ ] Task: Implement `CandidateEvaluatorAgent` in `src/problem_2_v2/initialization/evaluator.py`
    - [ ] Implement Pydantic AI candidate generation agent with Figure 10 prompt
    - [ ] Implement code fence extraction and AST validation
    - [ ] Integrate execution with `SubprocessRunner` and `DebuggerAgent`
    - [ ] Implement ranking and candidate sorting
    - [ ] Verify evaluator tests pass
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 4: Sequential Model Merging ($\mathcal{A}_{\text{merger}}$) & Pipeline Orchestrator
- [ ] Task: Write failing tests for ModelMergerAgent and InitializationPipeline
    - [ ] Create `tests/initialization/test_merger.py` testing merging prompt, blending code generation, and greedy acceptance
    - [ ] Create `tests/initialization/test_pipeline.py` testing end-to-end initialization workflow
- [ ] Task: Implement `ModelMergerAgent` in `src/problem_2_v2/initialization/merger.py`
    - [ ] Implement Pydantic AI merging agent with Figure 11 prompt
    - [ ] Implement greedy sequential loop (Algorithm 1) with score comparison via `MetricDirection`
    - [ ] Verify merger tests pass
- [ ] Task: Implement `InitializationPipeline` coordinator in `src/problem_2_v2/initialization/pipeline.py`
    - [ ] Wire TaskExtractor -> Retriever -> Evaluator -> Merger into single callable pipeline
    - [ ] Generate initial `PipelineArtifact` with complete lineage history
    - [ ] Verify pipeline integration tests pass
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)
