# Specification: Refinement Patch Auto-Indentation, Coder Self-Healing & Ablation Output Standardization

## Overview
This track addresses three critical reliability gaps in the Stage 2 (Targeted Refinement) pipeline:
1. **Indentation Collisions in `patch_script`:** When `CoderAgent` refines an indented target code block (e.g. inside a class or function), `patch_script` and `TargetCodeBlock.replace_in` either double-indent or fail to indent the replacement, causing `SyntaxError: unexpected indent` / `unindent does not match any outer indentation level` and resulting in `Score: n/a (Δ n/a)`.
2. **Missing Self-Repair on Patching Failures:** Because `patch_script` was called before `execution.run`, syntax errors caused by patching failed immediately without triggering `CoderAgent` retry feedback or falling back to `DebuggerAgent`.
3. **Inconsistent Ablation Outputs Across Iterations:** LLM-generated ablation scripts use arbitrary, non-standardized print formatting across outer iterations ($t=0, 1, 2$), confusing sign conventions (`delta` vs `drop`) and breaking downstream regex and summarizer parsing.

---

## Functional Requirements

1. **Smart Indentation Auto-Alignment:**
   - In `src/problem_2_v2/contracts/refinement.py` (`TargetCodeBlock._replace_by_substring`) and `src/problem_2_v2/refinement/coder.py` (`patch_script`):
     - Strip common leading whitespace from the replacement code using `textwrap.dedent()`.
     - Detect the exact base indentation level of the target block within the script.
     - Re-indent all lines of the replacement block to match the target base indentation before insertion.
     - Validate that the patched script parses as valid Python syntax via AST.

2. **Two-Layer Self-Healing in Inner Refinement Steps:**
   - In `RefinementPipeline._inner_step` (`src/problem_2_v2/refinement/pipeline.py`):
     - **Layer 1 (Agentic Coder Retry):** If `patch_script` fails due to syntax/indentation errors, feed the syntax error message back to `CoderAgent` for a retry attempt.
     - **Layer 2 (Debugger Fallback):** If patching still produces invalid syntax, pass the raw candidate script into `DebuggerAgent` so the debugger can repair the full script rather than discarding the inner step and recording `validation_score = None`.

3. **Standardized Ablation Output Contract:**
   - In `src/problem_2_v2/refinement/ablation.py`:
     - Update `_ABLATION_INSTRUCTIONS` to mandate exact tagged outputs:
       - `print(f"[ABLATION_BASELINE] score={baseline_score:.6f}")`
       - `print(f"[ABLATION_VARIANT] name={variant_name} score={variant_score:.6f} delta={delta:+.6f}")`
       - `print(f"[ABLATION_BEST] name={best_name} score={best_score:.6f} delta={best_delta:+.6f}")`
     - Explicitly define $\Delta = \text{Variant Score} - \text{Baseline Score}$ and categorize components into bottlenecks ($\Delta > 0$) vs essential ($\Delta < 0$).
     - Update `AblationSummarizerAgent._heuristic_report` to parse tagged formats with primary priority.

---

## Non-Functional Requirements
- **Zero Silent Dropped Iterations:** Inner refinement loops must execute and evaluate in the sandbox without dropping out to `Score: n/a`.
- **Quality Gates:** 100% pass rate on `uv run pytest`, `uv run ruff check src tests`, and `uv run mypy src tests`.

---

## Acceptance Criteria
- [ ] Patching replacements into class methods, functions, and nested blocks succeeds without indentation syntax errors.
- [ ] `_inner_step` recovers from initial coder syntax mistakes and executes in the sandbox.
- [ ] Ablation scripts output standardized tags (`[ABLATION_BASELINE]`, `[ABLATION_VARIANT]`) that parse deterministically across all outer iterations.
