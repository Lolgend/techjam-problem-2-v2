# Implementation Plan: Ingestion & Search-Guided Initialization Phase

## Phase 1: Task Ingestion & Search Provider Layer [checkpoint: 7f7d850]
- [x] Task: Write failing tests for TaskExtractor and SearchProviders (e75fce8)
    - [x] Create `tests/ingestion/test_extractor.py` testing markdown problem description extraction to `TaskSpecification`
    - [x] Create `tests/search/test_providers.py` testing `MockSearchProvider`, `TavilySearchProvider`, and `DuckDuckGoSearchProvider`
- [x] Task: Implement `TaskExtractor` agent in `src/problem_2_v2/ingestion/extractor.py` (f7f8302)
    - [x] Implement Pydantic AI agent with structured output bound to `TaskSpecification`
    - [x] Implement fallback heuristic markdown extractor for offline/mock environments
    - [x] Verify ingestion tests pass
- [x] Task: Implement `SearchProvider` interfaces in `src/problem_2_v2/search/providers.py` (f7f8302)
    - [x] Define `SearchProvider` protocol and `SearchResult` model
    - [x] Implement `MockSearchProvider`, `TavilySearchProvider`, `GoogleSearchProvider`, and `DuckDuckGoSearchProvider`
    - [x] Verify provider tests pass
- [x] Task: Write failing tests for `RetrieverAgent` (36bf108)
    - [x] Create `tests/search/test_retriever.py` testing query construction, model card parsing, and candidate count
- [x] Task: Implement `RetrieverAgent` in `src/problem_2_v2/search/retriever.py` (7f7d850)
    - [x] Implement query generation from `TaskSpecification`
    - [x] Implement Pydantic AI retriever agent with Figure 9 prompt to output `RetrievedCandidates`
    - [x] Verify retriever tests pass
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Execution Sandbox & Autonomous Debugger
- [x] Task: Write failing tests for SubprocessRunner (3d796cd)
    - [x] Create `tests/runner/test_sandbox.py`
    - [x] Test isolated sandbox directory creation and input data symlinking/copying
    - [x] Test execution timeout, exit code capture, and metric parsing (`ExecutionResult`)
    - [x] Test syntax error and exception capture
- [x] Task: Implement `SubprocessRunner` in `src/problem_2_v2/runner/sandbox.py` (3d95ead)
    - [x] Implement directory workspace management and `./input` mapping
    - [x] Implement subprocess execution with timeout and process termination
    - [x] Implement stdout/stderr capture and validation score regex extraction
    - [x] Verify sandbox tests pass
- [x] Task: Write failing tests for DebuggerAgent (2207ba3)
    - [x] Create `tests/runner/test_debugger.py` testing automated traceback repair for syntax/import/runtime errors
- [x] Task: Implement `DebuggerAgent` in `src/problem_2_v2/runner/debugger.py` (209d12b)
    - [x] Implement Pydantic AI debugger agent with Figure 19 prompt
    - [x] Implement iterative repair loop up to `max_debug_rounds`
    - [x] Verify debugger tests pass
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
