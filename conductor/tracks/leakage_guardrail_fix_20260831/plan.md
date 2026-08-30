# Plan: Leakage Guardrail Repair & Enforcement Fix

## Phase 1: Resilient Patching (`_patch()`) [checkpoint: c4e522c]

> Rewire `_patch()` to use `find_matching_block()` and add full-script rewrite fallback.

- [x] Task: Write failing tests for resilient patching (Red Phase) c4e522c
  - [x] Test `_patch()` succeeds with exact string match (existing behavior baseline)
  - [x] Test `_patch()` succeeds when suspicious block has minor whitespace/indent differences (fuzzy match via `find_matching_block`)
  - [x] Test `_patch()` succeeds when suspicious block has quote style differences (single vs double)
  - [x] Test `_patch()` succeeds via full-script rewrite fallback when fuzzy match also fails
  - [x] Test `_patch()` returns original code when full-script rewrite produces no extractable code
  - [x] Test `repair()` no longer raises `ValueError` — exhausts all tiers gracefully

- [x] Task: Implement resilient `_patch()` (Green Phase) c4e522c
  - [x] Import `find_matching_block` and `align_replacement_indent` in `leakage.py`
  - [x] Refactor `_patch()`: Tier 1 exact match → Tier 2 `find_matching_block()` + `align_replacement_indent()` → Tier 3 full-script rewrite via repair agent
  - [x] Add `_full_script_rewrite()` method that calls repair agent with full-script prompt
  - [x] Update `repair()` to pass `self` context for Tier 3 fallback
  - [x] Update `audit()` — remove `except ValueError` catch since `_patch()` no longer raises

- [x] Task: Refactor and verify (Refactor Phase) c4e522c
  - [x] Remove dead `block_in_script` import if no longer used directly
  - [x] Ensure all new tests pass
  - [x] Run `uv run pytest --cov=src/problem_2_v2/guardrails --cov-report=term-missing`

- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md) c4e522c

## Phase 2: Retry Loop & Config Fields [checkpoint: 8b113af]

> Add `max_leakage_retries` and the check→repair→re-check loop in `guard()`.

- [x] Task: Write failing tests for retry config and loop (Red Phase) 8b113af
  - [x] Test `ExecutionConfig` accepts `max_leakage_retries` with default `5`
  - [x] Test retry loop exits early when re-check confirms code is clean after 1st repair
  - [x] Test retry loop runs up to `max_leakage_retries` times when leakage persists
  - [x] Test retry loop passes latest repaired code to each subsequent `audit()` call
  - [x] Test `max_leakage_retries=0` skips retry entirely (single attempt, current behavior)

- [x] Task: Implement retry loop (Green Phase) 8b113af
  - [x] Add `max_leakage_retries: int = Field(default=5)` to `ExecutionConfig`
  - [x] Refactor `guard()` leakage section: wrap `audit()` in a `for` loop up to `max_leakage_retries`
  - [x] On each iteration, feed the latest `guarded` code back through `audit()`
  - [x] Break early when `status.is_leaking is False`
  - [x] Track `retries_used` counter for observability (Phase 4)

- [x] Task: Refactor and verify (Refactor Phase) 8b113af
  - [x] Extract retry logic into a private `_leakage_guard_loop()` method for readability
  - [x] Ensure all new and existing tests pass
  - [x] Run coverage check

- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md) 8b113af

## Phase 3: Strict Enforcement Mode [checkpoint: 8b113af]

> Add `strict_leakage` config and `LeakageEnforcementError`.

- [x] Task: Write failing tests for strict enforcement (Red Phase) 8b113af
  - [x] Test `ExecutionConfig` accepts `strict_leakage` with default `False`
  - [x] Test `strict_leakage=False` + exhausted retries → `guard()` warns and returns code (backward compatible)
  - [x] Test `strict_leakage=True` + exhausted retries → `guard()` raises `LeakageEnforcementError`
  - [x] Test `strict_leakage=True` + repair succeeds → `guard()` returns repaired code (no error)
  - [x] Test `LeakageEnforcementError` is a `RuntimeError` subclass with descriptive message

- [x] Task: Implement strict enforcement (Green Phase) 8b113af
  - [x] Define `LeakageEnforcementError(RuntimeError)` in `guardrails/leakage.py`
  - [x] Add `strict_leakage: bool = Field(default=False)` to `ExecutionConfig`
  - [x] After retry loop exhaustion in `guard()`: if `strict_leakage`, raise `LeakageEnforcementError`; otherwise warn and continue

- [x] Task: Refactor and verify (Refactor Phase) 8b113af
  - [x] Ensure error message includes retry count and last status details
  - [x] Ensure all tests pass
  - [x] Run coverage check

- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md) 8b113af

## Phase 4: Unambiguous Observability [checkpoint: 8b113af]

> Replace the ambiguous `execution.leakage_detected` with distinct events.

- [x] Task: Write failing tests for new Logfire events (Red Phase) 8b113af
  - [x] Test `repair()` emits `guardrails.leakage_repair.succeeded` on successful patching
  - [x] Test `repair()` emits `guardrails.leakage_repair.failed` when all tiers fail (existing event, verify kept)
  - [x] Test `guard()` emits `execution.leakage_repaired` with `retries_used` attribute when repair succeeds
  - [x] Test `guard()` emits `execution.leakage_unrepaired` with `retries_used` and `strict` attributes when repair fails
  - [x] Test `execution.leakage_detected` event is no longer emitted (removed)

- [x] Task: Implement new Logfire events (Green Phase) 8b113af
  - [x] In `leakage.py` `repair()`: add `logfire.info("guardrails.leakage_repair.succeeded")` on success path
  - [x] In `pipeline.py` `guard()`: replace `execution.leakage_detected` with `execution.leakage_repaired` or `execution.leakage_unrepaired` based on final outcome
  - [x] Include `retries_used` and `strict` as span attributes on all new events

- [x] Task: Refactor and verify (Refactor Phase) 8b113af
  - [x] Verify no remaining references to the old `execution.leakage_detected` event
  - [x] Ensure all tests pass including existing test suite
  - [x] Run full coverage: `uv run pytest --cov=src --cov-report=term-missing`

- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md) 8b113af
