# Implementation Plan: Ablation Sandbox Collision & Code Extraction Resiliency Fix

## Phase 1: Sandbox Preparation Hard-Link Idempotency (TDD) `[checkpoint: 135f490]`
- [x] Task: Write failing unit tests for `prepare_sandbox` hard-link idempotency in `tests/runner/test_sandbox.py` (`135f490`)
    - [x] Test repeated `prepare_sandbox` calls with identical dataset files does not raise `SameFileError`
    - [x] Test `prepare_sandbox` properly replaces modified target files without collisions
    - [x] Verify tests fail as expected (Red phase)
- [x] Task: Implement hard-link idempotency and iteration-scoped sandboxes (`135f490`)
    - [x] Update `SubprocessRunner.prepare_sandbox` in `src/problem_2_v2/runner/sandbox.py` to handle existing files, samefile checks, and `SameFileError`
    - [x] Update `AblationSummarizerAgent.summarize` in `src/problem_2_v2/refinement/ablation.py` and `src/problem_2_v2/refinement/pipeline.py` to scope ablation sandboxes (`sandbox_ablation_t{t}`)
    - [x] Verify sandbox unit tests pass (Green phase)
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Multi-Tier Resilient Code Block Extraction (TDD)
- [ ] Task: Write failing unit tests for resilient block extraction in `tests/refinement/test_extractor.py` and `tests/contracts/test_search_refinement.py`
    - [ ] Test `find_matching_block` with mismatched single/double quotes, trailing comments, and indentation
    - [ ] Test `find_matching_block` with contiguous anchor subset lines ($\ge 2$ lines)
    - [ ] Test `CodeBlockExtractorAgent.extract` with AST fallback when LLM output differs slightly from solution
    - [ ] Verify tests fail as expected (Red phase)
- [ ] Task: Implement `find_matching_block` and AST fallback in `contracts/refinement.py` and `refinement/extractor.py`
    - [ ] Implement `find_matching_block` in `src/problem_2_v2/contracts/refinement.py`
    - [ ] Update `CodeBlockExtractorAgent.extract` in `src/problem_2_v2/refinement/extractor.py` to use `find_matching_block` and fallback AST component extraction
    - [ ] Verify all extractor, refinement, and full test suite tests pass (Green phase)
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)
