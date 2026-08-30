# Implementation Plan: Baseline Evaluation & Submission Integration

## Phase 1: Task Specification & Sandbox Environment Plumbing (TDD) `[checkpoint: 86b34f9]`
- [x] Task: Write failing unit tests for sandbox PYTHONPATH injection and baseline module accessibility in `tests/test_sandbox.py` (86b34f9)
- [x] Task: Implement PYTHONPATH injection in `SubprocessRunner.run_code` in `src/problem_2_v2/runner/sandbox.py` (86b34f9)
- [x] Task: Verify task specification dry-run with updated `KuaiRand-Pure.md` (86b34f9)
- [x] Task: Phase 1 Verification & Checkpoint (Refer to workflow.md) (86b34f9)

## Phase 2: Agent Prompt Alignment & Submission Formatting (TDD)
- [x] Task: Write failing unit tests for `FinalArtifactProducer` and `CandidateEvaluatorAgent` prompts in `tests/test_finalizer.py` and `tests/test_evaluator.py` (a570d07)
- [x] Task: Update agent prompts in `evaluator.py`, `coder.py`, and `finalizer.py` to mandate `evaluate.py` and `submit.py` (a570d07)
- [ ] Task: Phase 2 Verification & Checkpoint (Refer to workflow.md)

## Phase 3: CLI Submission Verification & Full Integration
- [ ] Task: Write failing unit tests for CLI automated submission check in `tests/test_cli.py`
- [ ] Task: Implement automated submission verification with `submit.py --check` in `src/problem_2_v2/cli.py`
- [ ] Task: Run full test suite and verify end-to-end integration
- [ ] Task: Phase 3 Verification & Checkpoint (Refer to workflow.md)
