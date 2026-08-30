# Implementation Plan: Refinement Patch Auto-Indentation, Coder Self-Healing & Ablation Output Standardization

## Phase 1: Smart Indentation Alignment in Script Patching (TDD)
- [ ] Task: Write failing unit tests for indentation-aware patching in `tests/refinement/test_coder.py` and `tests/contracts/test_refinement.py`
    - [ ] Test replacing methods inside classes with unindented replacement code
    - [ ] Test replacing methods inside classes with pre-indented replacement code
    - [ ] Test replacing loop/nested blocks with mixed indentation
    - [ ] Verify tests fail as expected (Red phase)
- [ ] Task: Implement smart indentation normalization in `patch_script` and `TargetCodeBlock.replace_in`
    - [ ] Update `src/problem_2_v2/contracts/refinement.py` (`TargetCodeBlock._replace_by_substring`) to dedent and re-indent
    - [ ] Update `src/problem_2_v2/refinement/coder.py` (`patch_script`) with base-indentation alignment
    - [ ] Verify unit tests pass (Green phase)
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Inner Step Coder Self-Repair & Debugger Fallback (TDD)
- [ ] Task: Write failing unit tests for coder retry and debugger fallback in `tests/refinement/test_pipeline.py`
    - [ ] Test `_inner_step` retries with error feedback when `patch_script` or `coder` produces a syntax error
    - [ ] Test `_inner_step` falls back to `DebuggerAgent` on persistent syntax errors instead of returning `Score: n/a`
    - [ ] Verify tests fail as expected (Red phase)
- [ ] Task: Implement self-healing loop in `RefinementPipeline._inner_step`
    - [ ] Update `src/problem_2_v2/refinement/pipeline.py` to add coder retry feedback and debugger fallback
    - [ ] Verify unit tests pass (Green phase)
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 3: Standardized Ablation Output Contract (TDD)
- [ ] Task: Write failing unit tests for tagged ablation outputs in `tests/refinement/test_ablation.py`
    - [ ] Test parsing `[ABLATION_BASELINE]`, `[ABLATION_VARIANT]`, and `[ABLATION_BEST]` tags
    - [ ] Test sign convention handling (positive delta for bottlenecks, negative for essential components)
    - [ ] Verify tests fail as expected (Red phase)
- [ ] Task: Implement standardized prompt contract and tagged parser
    - [ ] Update `_ABLATION_INSTRUCTIONS` in `src/problem_2_v2/refinement/ablation.py`
    - [ ] Update `AblationSummarizerAgent._heuristic_report` to parse tagged formats
    - [ ] Verify all ablation and full suite tests pass (Green phase)
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)
