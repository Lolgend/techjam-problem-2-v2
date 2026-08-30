# Plan: Leakage Guardrail Repair & Enforcement Fix

## Phase 1: Resilient Patching (`_patch()`)

> Rewire `_patch()` to use `find_matching_block()` and add full-script rewrite fallback.

- [ ] Task: Write failing tests for resilient patching (Red Phase)
  - [ ] Test `_patch()` succeeds with exact string match (existing behavior baseline)
  - [ ] Test `_patch()` succeeds when suspicious block has minor whitespace/indent differences (fuzzy match via `find_matching_block`)
  - [ ] Test `_patch()` succeeds when suspicious block has quote style differences (single vs double)
  - [ ] Test `_patch()` succeeds via full-script rewrite fallback when fuzzy match also fails
  - [ ] Test `_patch()` returns original code when full-script rewrite produces no extractable code
  - [ ] Test `repair()` no longer raises `ValueError` — exhausts all tiers gracefully

- [ ] Task: Implement resilient `_patch()` (Green Phase)
  - [ ] Import `find_matching_block` and `align_replacement_indent` in `leakage.py`
  - [ ] Refactor `_patch()`: Tier 1 exact match → Tier 2 `find_matching_block()` + `align_replacement_indent()` → Tier 3 full-script rewrite via repair agent
  - [ ] Add `_full_script_rewrite()` method that calls repair agent with full-script prompt
  - [ ] Update `repair()` to pass `self` context for Tier 3 fallback
  - [ ] Update `audit()` — remove `except ValueError` catch since `_patch()` no longer raises

- [ ] Task: Refactor and verify (Refactor Phase)
  - [ ] Remove dead `block_in_script` import if no longer used directly
  - [ ] Ensure all new tests pass
  - [ ] Run `uv run pytest --cov=src/problem_2_v2/guardrails --cov-report=term-missing`

- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Retry Loop & Config Fields

> Add `max_leakage_retries` and the check→repair→re-check loop in `guard()`.

- [ ] Task: Write failing tests for retry config and loop (Red Phase)
  - [ ] Test `ExecutionConfig` accepts `max_leakage_retries` with default `5`
  - [ ] Test retry loop exits early when re-check confirms code is clean after 1st repair
  - [ ] Test retry loop runs up to `max_leakage_retries` times when leakage persists
  - [ ] Test retry loop passes latest repaired code to each subsequent `audit()` call
  - [ ] Test `max_leakage_retries=0` skips retry entirely (single attempt, current behavior)

- [ ] Task: Implement retry loop (Green Phase)
  - [ ] Add `max_leakage_retries: int = Field(default=5)` to `ExecutionConfig`
  - [ ] Refactor `guard()` leakage section: wrap `audit()` in a `for` loop up to `max_leakage_retries`
  - [ ] On each iteration, feed the latest `guarded` code back through `audit()`
  - [ ] Break early when `status.is_leaking is False`
  - [ ] Track `retries_used` counter for observability (Phase 4)

- [ ] Task: Refactor and verify (Refactor Phase)
  - [ ] Extract retry logic into a private `_leakage_guard_loop()` method for readability
  - [ ] Ensure all new and existing tests pass
  - [ ] Run coverage check

- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 3: Strict Enforcement Mode

> Add `strict_leakage` config and `LeakageEnforcementError`.

- [ ] Task: Write failing tests for strict enforcement (Red Phase)
  - [ ] Test `ExecutionConfig` accepts `strict_leakage` with default `False`
  - [ ] Test `strict_leakage=False` + exhausted retries → `guard()` warns and returns code (backward compatible)
  - [ ] Test `strict_leakage=True` + exhausted retries → `guard()` raises `LeakageEnforcementError`
  - [ ] Test `strict_leakage=True` + repair succeeds → `guard()` returns repaired code (no error)
  - [ ] Test `LeakageEnforcementError` is a `RuntimeError` subclass with descriptive message

- [ ] Task: Implement strict enforcement (Green Phase)
  - [ ] Define `LeakageEnforcementError(RuntimeError)` in `guardrails/leakage.py`
  - [ ] Add `strict_leakage: bool = Field(default=False)` to `ExecutionConfig`
  - [ ] After retry loop exhaustion in `guard()`: if `strict_leakage`, raise `LeakageEnforcementError`; otherwise warn and continue

- [ ] Task: Refactor and verify (Refactor Phase)
  - [ ] Ensure error message includes retry count and last status details
  - [ ] Ensure all tests pass
  - [ ] Run coverage check

- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 4: Unambiguous Observability

> Replace the ambiguous `execution.leakage_detected` with distinct events.

- [ ] Task: Write failing tests for new Logfire events (Red Phase)
  - [ ] Test `repair()` emits `guardrails.leakage_repair.succeeded` on successful patching
  - [ ] Test `repair()` emits `guardrails.leakage_repair.failed` when all tiers fail (existing event, verify kept)
  - [ ] Test `guard()` emits `execution.leakage_repaired` with `retries_used` attribute when repair succeeds
  - [ ] Test `guard()` emits `execution.leakage_unrepaired` with `retries_used` and `strict` attributes when repair fails
  - [ ] Test `execution.leakage_detected` event is no longer emitted (removed)

- [ ] Task: Implement new Logfire events (Green Phase)
  - [ ] In `leakage.py` `repair()`: add `logfire.info("guardrails.leakage_repair.succeeded")` on success path
  - [ ] In `pipeline.py` `guard()`: replace `execution.leakage_detected` with `execution.leakage_repaired` or `execution.leakage_unrepaired` based on final outcome
  - [ ] Include `retries_used` and `strict` as span attributes on all new events

- [ ] Task: Refactor and verify (Refactor Phase)
  - [ ] Verify no remaining references to the old `execution.leakage_detected` event
  - [ ] Ensure all tests pass including existing test suite
  - [ ] Run full coverage: `uv run pytest --cov=src --cov-report=term-missing`

- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)
