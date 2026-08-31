# Implementation Plan: Pydantic AI Built-in WebSearch Capability for RetrieverAgent

## Phase 1: Test-Driven Capability & Adaptive Retrieval Design (TDD Red Phase)
- [x] Task: Write unit tests for WebSearch capability integration and adaptive fallback (4636d5b)
  - [x] Add unit tests in `tests/search/test_retriever.py` testing `RetrieverAgent` initialization with `WebSearch()` capability
  - [x] Add unit tests verifying dynamic search agent invocation and candidate distillation with mock agent responses
  - [x] Add unit tests verifying graceful fallback to `SearchProvider` when explicitly provided or when dynamic search is unavailable
  - [x] Run test suite and confirm new tests fail as expected (Red Phase)
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Retriever Implementation with Pydantic AI WebSearch (TDD Green Phase)
- [x] Task: Refactor `RetrieverAgent` to support Pydantic AI `WebSearch` capability (ff72669)
  - [x] Import and configure `WebSearch` from `pydantic_ai.capabilities`
  - [x] Update `RetrieverAgent.__init__` to accept optional capabilities and optional search provider, defaulting to `capabilities=[WebSearch()]`
  - [x] Update `retrieve()` to use dynamic agent execution with autonomous search when `WebSearch` is active, while retaining `SearchProvider` snippet formatting when an explicit search provider is passed
  - [x] Ensure text fallback (`_text_fallback`) and domain starter cards (`_DOMAIN_FALLBACKS`) continue to guarantee non-empty candidate lists
  - [x] Run tests to confirm all tests pass (Green Phase)
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 3: Integration, Pipeline Verification & Documentation
- [x] Task: Verify downstream pipelines and end-to-end integration (2a1a8ec)
  - [x] Test `InitializationPipeline` integration with the updated `RetrieverAgent`
  - [x] Verify CLI dry-run and pipeline orchestrator functionality
  - [x] Run full test suite (`uv run pytest`) and check code coverage (>80%)
  - [x] Update `conductor/tech-stack.md` to document the Pydantic AI WebSearch capability adoption
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)
