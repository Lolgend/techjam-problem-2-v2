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

## Phase 3: Search Retrieval, Refinement & Guardrail Schemas [checkpoint: 3927a13]
- [x] Task: Write failing tests for Search, Ablation, Refinement, and Guardrail schemas (22e651c)
    - [x] Create `tests/contracts/test_search_refinement.py`
    - [x] Test `ModelCard` and `RetrievedCandidates` validation and code extraction
    - [x] Test `AblationVariant`, `AblationReport`, `TargetCodeBlock.replace_in`, and `RefinementPlan`
    - [x] Create `tests/contracts/test_guardrails.py`
    - [x] Test `DataLeakageStatus` prompt response parser ('Yes Data Leakage' / 'No Data Leakage')
    - [x] Test `DataUsageStatus` and `EnsembleStrategy`
- [x] Task: Implement Search & Candidate schemas in `src/problem_2_v2/contracts/search.py` (3927a13)
    - [x] Implement `ModelCard` and `RetrievedCandidates`
- [x] Task: Implement Refinement & Ablation schemas in `src/problem_2_v2/contracts/refinement.py` (3927a13)
    - [x] Implement `AblationVariant`, `AblationResultItem`, `AblationReport`
    - [x] Implement `TargetCodeBlock` with AST-safe `replace_in` helper
    - [x] Implement `RefinementPlan`
- [x] Task: Implement Guardrail schemas in `src/problem_2_v2/contracts/guardrails.py` (3927a13)
    - [x] Implement `DataLeakageStatus` with paper-exact prompt string normalization
    - [x] Implement `DataUsageStatus` and `EnsembleStrategy`
    - [x] Verify all tests pass
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 4: Unified Contract API, Quality Gate & Package Exports [checkpoint: dcc7586]
- [x] Task: Write failing integration and full serialization tests (90fc47f)
    - [x] Create `tests/contracts/test_integration.py`
    - [x] Test round-trip JSON serialization across all contract models
    - [x] Test type validation and immutability settings
- [x] Task: Package contract exports in `src/problem_2_v2/contracts/__init__.py` (dcc7586)
    - [x] Export all models, enums, and utility functions
    - [x] Run `uv run ruff check src tests` and `uv run ruff format src tests`
    - [x] Run `uv run mypy src`
    - [x] Run `uv run pytest --cov=src --cov-report=term-missing` (>80% coverage check)
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)
