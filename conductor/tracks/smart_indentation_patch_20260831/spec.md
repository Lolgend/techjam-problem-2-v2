# Specification: Smart Indentation Auto-Alignment & Coder Self-Healing in Script Patching

## Overview
This track fixes inner refinement step failures where script patching crashes due to indentation or syntax errors, causing the pipeline to silently swallow the error, burn through inner loop attempts, and output `Score: n/a (Δ n/a)`.

---

## Functional Requirements

1. **Smart Indentation Auto-Alignment (`align_replacement_indent`):**
   - In `src/problem_2_v2/contracts/refinement.py`:
     - Detect the base indentation (leading whitespace) of the first non-blank line of the matched target code in the script.
     - Dedent the replacement string with `textwrap.dedent()` to normalize its relative indentation.
     - Re-indent every non-blank line of the dedented replacement string to match the target's base indentation level.
   - In `TargetCodeBlock._replace_by_substring`:
     - Apply `align_replacement_indent` before substituting the block into `full_script`.
   - In `src/problem_2_v2/refinement/coder.py`:
     - Update `patch_script` to route all patching through `TargetCodeBlock.replace_in`.

2. **In-Step Coder Self-Repair Loop:**
   - In `src/problem_2_v2/refinement/coder.py`:
     - Add `CoderAgent.repair(target_block, plan, invalid_code, error_message)` to re-prompt the LLM with exact syntax/indentation error feedback if an initial block fails validation.
   - In `src/problem_2_v2/refinement/pipeline.py` (`RefinementPipeline._inner_step`):
     - When `patch_script` fails due to a `ValueError` (syntax or patching failure), invoke `self.coder.repair(...)` for 1 retry round within the current inner step.

3. **Full-Script Debugger Fallback:**
   - If patching still fails after in-step repair, pass the best-effort stitched candidate to `self.debugger.debug(...)` to let `DebuggerAgent` repair the full script rather than failing silently and returning `Score: n/a`.

---

## Non-Functional Requirements
- **Zero Silent Dropped Iterations:** Inner refinement attempts must never be silently discarded due to patching or syntax errors.
- **Quality Gates:** 100% pass rate on `uv run pytest`, `uv run ruff check src tests`, and `uv run mypy src tests`.

---

## Acceptance Criteria
- [ ] Patching unindented or pre-indented replacement code into class methods, functions, and nested blocks succeeds without syntax errors.
- [ ] In-step coder self-repair successfully fixes initial syntax errors without consuming subsequent inner loop attempts.
- [ ] `DebuggerAgent` successfully recovers malformed stitched scripts as a fallback.
- [ ] All unit tests in `tests/refinement/test_coder.py`, `tests/contracts/test_refinement.py`, and `tests/refinement/test_pipeline.py` pass.
