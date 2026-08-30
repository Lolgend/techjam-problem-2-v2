# Implementation Plan: Smart Indentation Auto-Alignment & Coder Self-Healing in Script Patching

## Phase 1: Smart Indentation Alignment in Script Patching (TDD) `[checkpoint: 2dfb643]`
- [x] Task: Write failing unit tests for indentation-aware patching in `tests/refinement/test_coder.py` and `tests/contracts/test_search_refinement.py` (`2dfb643`)
    - [x] Test replacing methods inside classes with unindented replacement code
    - [x] Test replacing methods inside classes with pre-indented replacement code
    - [x] Test replacing loop/nested blocks with mixed relative indentations
    - [x] Verify tests fail as expected (Red phase)
- [x] Task: Implement smart indentation normalization in `patch_script` and `TargetCodeBlock.replace_in` (`2dfb643`)
    - [x] Implement `align_replacement_indent` and update `TargetCodeBlock._replace_by_substring` in `src/problem_2_v2/contracts/refinement.py`
    - [x] Update `patch_script` in `src/problem_2_v2/refinement/coder.py` to route all patching through `TargetCodeBlock.replace_in`
    - [x] Verify unit tests pass (Green phase)
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 2: In-Step Coder Self-Repair & Debugger Fallback (TDD) `[checkpoint: 40e721e]`
- [x] Task: Write failing unit tests for in-step coder repair and debugger fallback in `tests/refinement/test_pipeline.py` (`40e721e`)
    - [x] Test `_inner_step` retries with error feedback when `patch_script` or `coder` produces a syntax error
    - [x] Test `_inner_step` falls back to `DebuggerAgent` on persistent syntax errors instead of returning `Score: n/a`
    - [x] Verify tests fail as expected (Red phase)
- [x] Task: Implement self-healing loop in `CoderAgent` and `RefinementPipeline._inner_step` (`40e721e`)
    - [x] Add `repair()` method to `CoderAgent` in `src/problem_2_v2/refinement/coder.py`
    - [x] Update `_inner_step` in `src/problem_2_v2/refinement/pipeline.py` to add in-step coder retry and debugger fallback
    - [x] Verify unit tests pass (Green phase)
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)
