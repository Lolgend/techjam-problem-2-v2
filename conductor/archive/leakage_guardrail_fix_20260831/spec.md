# Spec: Leakage Guardrail Repair & Enforcement Fix

## Overview

The data leakage guardrail (`DataLeakageCheckerAgent`) detects leakage correctly but **fails silently** when patching the repair back into the script. The pipeline then executes the original leaky code without error or clear indication that repair failed. Two root causes:

1. **Brittle patching** — `_patch()` in `leakage.py` uses exact string matching (`str.replace`) and a basic `block_in_script` regex, while the codebase already has a robust multi-tier matcher (`find_matching_block`) in `contracts/refinement.py` that it doesn't use.
2. **No enforcement or retry** — `guard()` in `pipeline.py` logs `execution.leakage_detected` regardless of repair outcome and always continues execution — there is no retry loop and no strict-abort option.
3. **Ambiguous observability** — the Logfire trace cannot distinguish "detected and repaired" from "detected and repair failed" because both paths emit the same event.

## Affected Files

| File | Role |
|------|------|
| `src/problem_2_v2/guardrails/leakage.py` | `DataLeakageCheckerAgent` — check, repair, audit, `_patch()` |
| `src/problem_2_v2/execution/pipeline.py` | `ExecutionGuardrailPipeline` — `guard()`, `ExecutionConfig` |
| `src/problem_2_v2/contracts/guardrails.py` | `DataLeakageStatus` contract |
| `src/problem_2_v2/contracts/refinement.py` | `find_matching_block`, `align_replacement_indent` (existing utilities to reuse) |

## Functional Requirements

### FR-1: Resilient Patching (`_patch()`)

**Current behavior:** `_patch()` tries `str.replace()`, falls back to `block_in_script()` + `TargetCodeBlock.replace_in()`, and raises `ValueError` if neither works.

**Required behavior:**
1. **Tier 1 — Exact match:** Try `str.replace(original, corrected, 1)` (current).
2. **Tier 2 — Fuzzy/normalized match:** Use the existing `find_matching_block()` from `contracts/refinement.py` to locate the suspicious block with whitespace normalization, quote unification, comment stripping, and AST-level fallback. Replace the matched verbatim segment with the corrected block (aligned via `align_replacement_indent()`).
3. **Tier 3 — Full-script rewrite:** If fuzzy matching still fails, call the repair agent with a modified prompt asking it to rewrite the **entire script** (not just the suspicious block) with leakage fixed. Return the rewritten script directly.

The `_patch()` method should no longer raise `ValueError` on match failure — it should exhaust all tiers and only fail if the full-script rewrite also produces no code.

### FR-2: Retry Loop with Configurable Budget

**Current behavior:** `audit()` calls `check()` → `repair()` once. If repair fails, it returns the original code unchanged.

**Required behavior:**
- `guard()` in `ExecutionGuardrailPipeline` wraps the check→repair cycle in a retry loop of up to `max_leakage_retries` attempts (new `ExecutionConfig` field, default `5`).
- Each retry re-runs `audit()` on the latest code version (which may have been partially repaired).
- The loop exits early when `status.is_leaking` is `False` (repair succeeded and re-check confirms clean).
- If all retries are exhausted and leakage persists, behavior depends on `strict_leakage` (see FR-3).

### FR-3: Strict Enforcement Mode

**Current behavior:** `guard()` always continues execution after logging `execution.leakage_detected`.

**Required behavior:**
- New `ExecutionConfig` field: `strict_leakage: bool = False`.
- When `strict_leakage=True` and all retry attempts are exhausted with leakage still present, `guard()` raises a new `LeakageEnforcementError` (subclass of `RuntimeError`) instead of continuing.
- When `strict_leakage=False` (default — backward compatible), `guard()` warns and continues with the best-effort code, preserving current behavior.

### FR-4: Unambiguous Observability

**Current behavior:** A single `execution.leakage_detected` event fires whether repair succeeded or failed.

**Required behavior:**
- **Guardrail layer** (`leakage.py`):
  - On successful repair: emit `logfire.info("guardrails.leakage_repair.succeeded")`.
  - On failed repair (all tiers exhausted): emit `logfire.warn("guardrails.leakage_repair.failed")` (already exists, keep it).
  - On no-code from repair agent: emit `logfire.warn("guardrails.leakage_repair.no_code")` (already exists, keep it).
- **Pipeline layer** (`pipeline.py`):
  - Replace the single `execution.leakage_detected` event with two distinct events:
    - `logfire.info("execution.leakage_repaired")` — leakage was detected **and** successfully repaired.
    - `logfire.warn("execution.leakage_unrepaired")` — leakage was detected but repair failed after all retries.
  - Include attributes: `retries_used: int`, `strict: bool`.

## Non-Functional Requirements

- **Backward compatibility:** Default config values (`strict_leakage=False`, `max_leakage_retries=5`) must preserve existing behavior for users who don't opt in to strict mode.
- **Test coverage:** > 80% coverage for all modified modules.
- **No new dependencies:** All changes use existing libraries (`logfire`, `pydantic`, `pydantic-ai`).

## Acceptance Criteria

1. **AC-1:** When the check agent flags a suspicious block with minor whitespace/formatting differences from the actual script, `_patch()` successfully locates and replaces it using `find_matching_block()`.
2. **AC-2:** When fuzzy matching also fails, the repair agent is re-invoked with a full-script rewrite prompt, and the returned script replaces the original.
3. **AC-3:** The retry loop re-checks after each repair attempt and exits early when the code is confirmed clean.
4. **AC-4:** With `strict_leakage=True`, a `LeakageEnforcementError` is raised after exhausting all retries.
5. **AC-5:** With `strict_leakage=False`, execution continues with a warning (backward compatible).
6. **AC-6:** Logfire traces unambiguously show `execution.leakage_repaired` or `execution.leakage_unrepaired` with retry count.
7. **AC-7:** All existing tests continue to pass without modification.

## Out of Scope

- AST-level semantic equivalence checking of repaired code (beyond what `find_matching_block` already does).
- Changes to the repair agent's LLM prompt engineering (beyond the new full-script rewrite fallback prompt).
- Changes to the data usage guardrail (`DataUsageCheckerAgent`).
- Changes to the debugger loop or sandbox runner.
