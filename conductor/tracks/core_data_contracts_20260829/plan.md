# Implementation Plan: Core Architecture & Pydantic Data Contracts

## Phase 1: Code Utilities & Base Contract Infrastructure [checkpoint: 76fe8ed]
- [x] Task: Write failing unit tests for code extraction, AST syntax validation, and diff generation (1aa94f0)
    - [x] Create `tests/contracts/test_code_utils.py`
    - [x] Test markdown code block extraction (fenced with ```python, raw code, mixed text)
    - [x] Test AST syntax validation with valid code and syntax error cases
    - [x] Test unified diff computation
- [x] Task: Implement code utilities in `src/problem_2_v2/contracts/code_utils.py` (ade4ae4)
    - [x] Implement `extract_python_code`
    - [x] Implement `validate_python_syntax`
    - [x] Implement `compute_code_diff`
    - [x] Verify tests pass
- [x] Task: Write failing tests for core Enums and MetricDirection logic (a756c71)
    - [x] Create `tests/contracts/test_enums.py`
    - [x] Test `MetricDirection.is_better` and `MetricDirection.delta` for MAXIMIZE / MINIMIZE
    - [x] Test `TaskType` and `ComponentCategory` enum members
- [x] Task: Implement enums in `src/problem_2_v2/contracts/enums.py` (76fe8ed)
    - [x] Implement `MetricDirection`, `TaskType`, `ComponentCategory`, `EnsembleMethod`
    - [x] Verify tests pass
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Task Specification, Execution Telemetry & Artifact Lineage [checkpoint: 271d07b]
- [x] Task: Write failing tests for TaskSpecification markdown parser and ExecutionResult (d1b0228)
    - [x] Create `tests/contracts/test_task.py`
    - [x] Test markdown problem description parser (`from_markdown`) with sample competition specs
    - [x] Test ExecutionResult validation score regex extraction (`Final Validation Performance: {score}`)
    - [x] Test PipelineArtifact lineage tracking and JSON serialization
- [x] Task: Implement TaskSpecification, ExecutionResult, and PipelineArtifact in `src/problem_2_v2/contracts/task.py` (271d07b)
    - [x] Define `TaskSpecification` with markdown ingestion helper
    - [x] Define `ExecutionResult` with metric extraction and error traceback capture
    - [x] Define `PipelineArtifact` with code diff and version history
    - [x] Verify tests pass
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 3: Search Retrieval, Refinement & Guardrail Schemas
- [ ] Task: Write failing tests for Search, Ablation, Refinement, and Guardrail schemas
    - [ ] Create `tests/contracts/test_search_refinement.py`
    - [ ] Test `ModelCard` and `RetrievedCandidates` validation and code extraction
    - [ ] Test `AblationVariant`, `AblationReport`, `TargetCodeBlock.replace_in`, and `RefinementPlan`
    - [ ] Create `tests/contracts/test_guardrails.py`
    - [ ] Test `DataLeakageStatus` prompt response parser ('Yes Data Leakage' / 'No Data Leakage')
    - [ ] Test `DataUsageStatus` and `EnsembleStrategy`
- [ ] Task: Implement Search & Candidate schemas in `src/problem_2_v2/contracts/search.py`
    - [ ] Implement `ModelCard` and `RetrievedCandidates`
- [ ] Task: Implement Refinement & Ablation schemas in `src/problem_2_v2/contracts/refinement.py`
    - [ ] Implement `AblationVariant`, `AblationResultItem`, `AblationReport`
    - [ ] Implement `TargetCodeBlock` with AST-safe `replace_in` helper
    - [ ] Implement `RefinementPlan`
- [ ] Task: Implement Guardrail schemas in `src/problem_2_v2/contracts/guardrails.py`
    - [ ] Implement `DataLeakageStatus` with paper-exact prompt string normalization
    - [ ] Implement `DataUsageStatus` and `EnsembleStrategy`
    - [ ] Verify all tests pass
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 4: Unified Contract API, Quality Gate & Package Exports
- [ ] Task: Write failing integration and full serialization tests
    - [ ] Create `tests/contracts/test_integration.py`
    - [ ] Test round-trip JSON serialization across all contract models
    - [ ] Test type validation and immutability settings
- [ ] Task: Package contract exports in `src/problem_2_v2/contracts/__init__.py`
    - [ ] Export all models, enums, and utility functions
    - [ ] Run `uv run ruff check src tests` and `uv run ruff format src tests`
    - [ ] Run `uv run mypy src`
    - [ ] Run `uv run pytest --cov=src --cov-report=term-missing` (>80% coverage check)
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)
