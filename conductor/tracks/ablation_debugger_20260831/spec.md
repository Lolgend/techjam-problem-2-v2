# Specification: Integrate `DebuggerAgent` into Ablation Study Workflow

## Overview
Currently, `AblationAgent` generates an ablation study script, which `AblationSummarizerAgent` executes directly via `runner.run_code()`. If the generated ablation code crashes (e.g., syntax error, broken import, variable scoping bug, or runtime exception), `raw_output` receives the error traceback and passes it directly to the summarizer without any repair attempt.

This track updates `AblationSummarizerAgent` and `RefinementPipeline` to leverage `DebuggerAgent` to autonomously detect execution failures and iteratively repair failing ablation scripts (up to `max_debug_rounds`) before digesting the output into an `AblationReport`.

## Functional Requirements
1. **Debugger Injection**:
   - Update `AblationSummarizerAgent.__init__` to accept an optional `debugger: DebuggerAgent | None = None` (defaulting to a `DebuggerAgent` constructed with the provided runner and model).
   - Update `RefinementPipeline` and `MLEStarPipeline._build_branch` to pass the branch-scoped `DebuggerAgent` into `AblationSummarizerAgent`.
2. **Error Detection & Repair Loop**:
   - When executing `ablation_code` in `AblationSummarizerAgent.summarize()`, check if execution failed (non-zero return code, exception, or stderr traceback).
   - If execution fails and a `debugger` is available, invoke the debugger repair loop to iteratively fix the code and re-run in the ablation sandbox up to `max_debug_rounds`.
   - Ensure the debugger repair instructions for ablation preserve the multi-variant evaluation structure (e.g., testing 2–3 variants and printing each variant's performance).
3. **Summarization on Repaired Outcome**:
   - Pass the final execution result's output (`stdout` or `stderr`) from the debug outcome to the LLM summarizer or heuristic fallback.
   - Record the number of ablation debug repair rounds and events in Logfire spans.
4. **Resilience & Fallbacks**:
   - If repair exhausts all rounds without succeeding, gracefully fall back to the existing behavior: summarize the raw stderr/stdout or use `_heuristic_report` without halting the refinement outer loop.

## Acceptance Criteria
- Unit tests in `tests/refinement/test_ablation.py` verifying:
  - Failing ablation scripts trigger the debugger repair loop.
  - Successfully repaired ablation scripts produce valid `AblationReport` summaries.
  - Persistent failures gracefully fall back to raw output / heuristic parsing.
- All existing tests in `tests/refinement/` and the complete test suite (`uv run pytest`) pass with >80% coverage.
- Code adheres to strict type checks (`uv run mypy src`) and formatting (`uv run ruff check src`).
