# Implementation Plan: Baseline Evaluation & Submission Integration

## Phase 1: Task Specification & Sandbox Environment Plumbing (TDD) `[checkpoint: 7b0a207]`
- [x] Task: Write failing unit tests for sandbox PYTHONPATH injection and baseline module accessibility in `tests/test_sandbox.py` (7b0a207)
- [x] Task: Implement PYTHONPATH injection in `SubprocessRunner.run_code` in `src/problem_2_v2/runner/sandbox.py` (7b0a207)
- [x] Task: Update `KuaiRand-Pure.md` with explicit `evaluate.py` metric definition and `submit.py` format requirements (7b0a207)
- [x] Task: Phase 1 Verification & Checkpoint (Refer to workflow.md) (7b0a207)

## Phase 2: Finalizer Prompt Alignment & Submission Formatting (TDD) `[checkpoint: 3043abd]`
- [x] Task: Write failing unit tests for `FinalArtifactProducer` submission schema instructions in `tests/test_finalizer.py` (3043abd)
- [x] Task: Update `_FINALIZER_INSTRUCTIONS` and `build_prompt` in `src/problem_2_v2/execution/finalizer.py` to enforce `row_id,user_id,video_id,score` (3043abd)
- [x] Task: Phase 2 Verification & Checkpoint (Refer to workflow.md) (3043abd)

## Phase 3: CLI Submission Verification & Integration Testing
- [x] Task: Write failing unit tests for CLI submission check in `tests/test_cli.py` (67cae2c)
- [x] Task: Implement automated submission verification with `submit.py --check` in `src/problem_2_v2/cli.py` (67cae2c)
- [x] Task: Full system integration test with dry-run and mock submission validation (67cae2c)
- [ ] Task: Phase 3 Verification & Checkpoint (Refer to workflow.md)
