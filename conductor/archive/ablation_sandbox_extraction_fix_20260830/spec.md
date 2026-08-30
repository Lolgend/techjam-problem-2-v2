# Specification: Ablation Sandbox Collision & Code Extraction Resiliency Fix

## Overview
This track fixes two critical failure points in the Stage 2 (Targeted Refinement) pipeline:
1. **Windows `shutil.SameFileError` during repeated sandbox preparation:** In Outer iterations 2–5, `prepare_sandbox` with `candidate_id="ablation"` fails when attempting to re-link/copy hard-linked input files, raising `shutil.SameFileError` and crashing `_outer_step` (`refinement.outer.failed`).
2. **Code block extraction failure from minor LLM formatting mismatches:** When `CodeBlockExtractorAgent` extracts a code block, minor discrepancies in quote styles (`'` vs `"`), trailing comments, or indentation cause strict regex matching in `block_in_script` to fail with `ValueError: Extracted code block not found in solution script.` (`refinement.extract.failed`), discarding successful ablation discoveries and skipping inner optimization loops.

---

## Functional Requirements

1. **Idempotent Sandbox Preparation (`prepare_sandbox`):**
   - In `SubprocessRunner.prepare_sandbox`, before creating a hard link or copying:
     - Check if `target.exists()`.
     - If `source.samefile(target)` is true, skip copying/linking as the file is already mapped.
     - If `target.exists()` but is not the same file, remove/unlink `target` before re-linking or copying.
     - Add explicit exception handling for `shutil.SameFileError` and `FileExistsError` to guarantee idempotency.
   - Update `AblationSummarizerAgent.summarize` to accept an `iteration_index: int | None = None` and scope the sandbox directory to `sandbox_ablation_t{t}` (or keep `sandbox_ablation` safe).

2. **Multi-Tier Resilient Code Block Extraction (`find_matching_block`):**
   - Implement `find_matching_block(code_block: str, script: str) -> str | None` in `problem_2_v2.contracts.refinement`:
     - **Tier 1 (Verbatim / Indentation-tolerant):** Search using `block_in_script`.
     - **Tier 2 (Normalized Line Matching):** Match lines after normalizing string quotes (`'` <-> `"`), stripping inline comments, and trimming whitespace.
     - **Tier 3 (Contiguous Anchor Matching):** Match the longest contiguous sequence of non-blank lines ($\ge 2$ lines) found in `script`.
     - **Tier 4 (AST Definition Fallback):** If `code_block` contains a function/class header, parse `script` AST and extract the corresponding function/class definition node.
   - Update `CodeBlockExtractorAgent.extract`:
     - Use `find_matching_block(item.code_block, solution)` to extract the verbatim code block from `solution`.
     - If `find_matching_block` returns `None`, fall back to extracting the primary training/model/loss function block from `solution` (using AST parsing) rather than throwing `ValueError` and crashing the refinement loop.

---

## Non-Functional Requirements
- **Robustness & Fault Tolerance:** Refinement loops must never crash or skip inner iterations due to cosmetic LLM formatting differences or repeated sandbox mappings.
- **Coverage & Quality Gates:** Passes `uv run pytest`, `uv run mypy src tests`, and `uv run ruff check src tests` with >80% coverage on new code.
- **Zero Breaking Changes:** Preserves all existing method signatures and return models.

---

## Acceptance Criteria
- [ ] Repeated calls to `SubprocessRunner.prepare_sandbox` with hard links do not raise `shutil.SameFileError` or `FileExistsError`.
- [ ] `find_matching_block` successfully matches code blocks containing quote differences, comments, and partial line runs.
- [ ] `CodeBlockExtractorAgent.extract` successfully extracts a valid `TargetCodeBlock` even when the LLM modifies quotes or adds comments.
- [ ] End-to-end multi-outer refinement runs execute without `refinement.outer.failed` or `refinement.extract.failed`.
