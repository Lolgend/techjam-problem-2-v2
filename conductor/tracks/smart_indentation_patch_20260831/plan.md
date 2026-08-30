# Implementation Plan: Smart Indentation Auto-Alignment in Script Patching

## Phase 1: Smart Indentation Alignment in Script Patching (TDD)
- [ ] Task: Write failing unit tests for indentation-aware patching in `tests/refinement/test_coder.py` and `tests/contracts/test_refinement.py`
    - [ ] Test replacing methods inside classes with unindented replacement code
    - [ ] Test replacing methods inside classes with pre-indented replacement code
    - [ ] Test replacing loop/nested blocks with mixed relative indentations
    - [ ] Verify tests fail as expected (Red phase)
- [ ] Task: Implement smart indentation normalization in `patch_script` and `TargetCodeBlock.replace_in`
    - [ ] Implement `align_replacement_indent` and update `TargetCodeBlock._replace_by_substring` in `src/problem_2_v2/contracts/refinement.py`
    - [ ] Update `patch_script` in `src/problem_2_v2/refinement/coder.py` to route all patching through `TargetCodeBlock.replace_in`
    - [ ] Verify unit tests pass (Green phase)
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)
