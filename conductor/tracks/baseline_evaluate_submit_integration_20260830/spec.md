# Specification: Baseline Evaluation (`evaluate.py`) & Submission (`submit.py`) Integration

## Overview
This track integrates and mandates the official KuaiRand-Pure evaluation harness (`src/baseline/evaluate.py`) for all model validation and the official submission formatter/checker (`src/baseline/submit.py`) for production finalization in MLE-STAR.

## Functional Requirements
1. **Task Specification Mandate (`KuaiRand-Pure.md`):**
   - Mandate that all candidate generation, refinement, and ensembling scripts evaluate validation performance using `evaluate(val_user_ids, val_labels, val_predictions)` from `evaluate.py`.
   - Mandate that the production finalizer outputs `./final/submission.csv` with columns `row_id,user_id,video_id,score` adhering to `submit.py`.

2. **Sandbox Environment Plumbing (`SubprocessRunner`):**
   - Configure `SubprocessRunner.run_code` to inject `src/baseline` and project root into `PYTHONPATH`.
   - Ensure `from evaluate import evaluate` and `from submit import write_submission` can be imported in any candidate sandbox without error.

3. **Agent Prompt Alignment:**
   - Update `CandidateEvaluatorAgent`, `CoderAgent`, and `FinalArtifactProducer` instructions to instruct models to use `from evaluate import evaluate` and format submissions using `row_id,user_id,video_id,score`.

4. **CLI Automated Submission Verification (`cli.py`):**
   - Run automated validation with `submit.py --check` on the production submission file after pipeline completion.

## Non-Functional Requirements
- Maintain >80% test coverage for modified modules.
- Pass `mypy src` and `ruff check src`.
- Pass all unit tests in `pytest`.
