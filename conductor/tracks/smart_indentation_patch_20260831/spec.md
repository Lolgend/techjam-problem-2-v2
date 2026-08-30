# Specification: Smart Indentation Auto-Alignment in Script Patching

## Overview
This track fixes indentation syntax errors during Stage 2 (Targeted Refinement) script patching. When `CoderAgent` refines an indented target code block (e.g. inside a class method or training loop), `patch_script` and `TargetCodeBlock.replace_in` currently cause `SyntaxError: unexpected indent` or `unindent does not match any outer indentation level` when replacing code blocks.

---

## Functional Requirements

1. **`align_replacement_indent` Helper:**
   - In `src/problem_2_v2/contracts/refinement.py`:
     - Detect the base indentation (leading whitespace) of the first non-blank line of the matched target code in the script.
     - Dedent the replacement string with `textwrap.dedent()` to normalize its indentation.
     - Re-indent every non-blank line of the dedented replacement string to match the target's base indentation level.

2. **Integration with `TargetCodeBlock._replace_by_substring`:**
   - Use `align_replacement_indent` to format the replacement code before splicing it into `full_script`.

3. **Integration with `patch_script`:**
   - Update `patch_script` in `src/problem_2_v2/refinement/coder.py` to route all patching through `TargetCodeBlock.replace_in` so indentation normalization is always applied.

---

## Acceptance Criteria
- [ ] Patching unindented replacement code into an 8-space indented class method succeeds and parses as valid Python.
- [ ] Patching pre-indented replacement code (already possessing 8 spaces) does not double-indent to 16 spaces.
- [ ] All unit tests in `tests/refinement/test_coder.py` and `tests/contracts/test_refinement.py` pass.
