# Product Guidelines: MLE-STAR

## Core Philosophy & Design Principles

### 1. Reproducibility & Determinism
- Every generated script must enforce deterministic behavior with explicit random seeds across PyTorch, NumPy, scikit-learn, and tree-boosting libraries (LightGBM, XGBoost, CatBoost).
- Hold-out validation splits, 30k subsampling routines, and evaluation metrics must be verifiable and deterministic across runs.

### 2. Defensive Code Generation & Modularity
- Generated solutions must be standalone, single-file runnable Python scripts that do not rely on undeclared local modules or undefined globals.
- Scripts must strictly output the standardized evaluation line: `Final Validation Performance: {final_validation_score}`.
- Disallow empty `try-except` blocks or exception swallowers (`pass`) that mask runtime failures; ensure all exceptions generate rich tracebacks for the Debugging Agent ($A_{\text{debugger}}$).

### 3. Strict Data Leakage Prevention
- Feature transformers, scalers, encoders, and imputers must only be fitted on training partitions and applied to validation/test sets.
- All data preprocessing code segments undergo static analysis and LLM inspection ($A_{\text{leakage}}$) prior to execution.

### 4. Observability & Real-Time Telemetry
- Rich, formatted terminal output with live score progression tables, outer/inner loop iteration indicators, and clear phase transitions.
- Integrated Pydantic Logfire tracing for LLM prompts, model responses, code diffs, execution output, memory/time stats, and validation scores.

### 5. Artifact Management & Execution Safeguards
- **Complete Artifact Retention:** Every candidate script, ablation experiment, summarization log, and score trajectory is saved under a timestamped directory (`./runs/<run_id>/...`).
- **Execution Safeguards:** Local subprocess execution with configurable per-script timeout (default: 10 minutes), memory monitoring, CUDA GPU support, and graceful timeout/OOM trapping.
