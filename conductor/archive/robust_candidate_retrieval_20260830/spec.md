# Specification: Robust Multi-Provider Model Card Retrieval, Fallback Parsing & Baseline Seeding

## 1. Overview
Fixes the root cause of `Score: n/a` by implementing a resilient, multi-tiered candidate retrieval and evaluation pipeline. It provides dual-mode JSON/markdown regex extraction for `RetrieverAgent`, domain-aware fallback model cards for all task types, automatic baseline starter code injection from `src/baseline/`, and clear diagnostic traceback reporting.

## 2. Functional Requirements

### A. Resilient Dual-Mode Model Card Retrieval (`src/problem_2_v2/search/retriever.py`)
- **Dual-Mode Parsing:**
  1. Primary: Structured Pydantic AI output (`output_type=list[ModelCard]`).
  2. Text Fallback: JSON array extraction (`json.loads`) and markdown code block parser from raw response string when structured tool-calling is not supported by the LLM provider.
- **Domain-Aware Default Fallback:**
  - If retrieved cards is empty, automatically populate domain-specific starter architectures for the given `TaskType`:
    - `RECOMMENDER_RANKING`: Factorization Machine (BPR loss), DeepFM, LightGBM Ranker, Item Popularity.
    - `TABULAR_CLASSIFICATION` / `REGRESSION`: LightGBM, XGBoost, CatBoost, Multi-Layer Perceptron.
    - Other modalities: Standard domain baseline architectures.

### B. Official Baseline Starter Code Injection (`src/problem_2_v2/initialization/pipeline.py`)
- Check for starter script files in `src/baseline/baseline.py`, `baseline.py`, or `TaskSpecification.dataset_dir`.
- If available, evaluate the starter code as an official candidate card alongside retrieved models, ensuring $s_0$ achieves at least the baseline validation score ($0.6016$).

### C. Diagnostic Reporting & Robust Merging (`initialization/evaluator.py`, `initialization/merger.py`)
- Print candidate generation and execution status to the terminal (`[Candidate {i}/{M}] {name} -> Score: {score}`).
- If a candidate script fails in the sandbox, log the error reason to console.
- In `ModelMergerAgent`, if all merge attempts fail or are rejected, preserve the best individual candidate script rather than returning empty code.

## 3. Non-Functional Requirements
- **Fault-Tolerant:** Never returns empty candidates regardless of network hiccups or non-standard LLM text formatting.
- **Zero Regressions:** 100% compatibility across all 310 existing unit and integration tests.

## 4. Acceptance Criteria
- [ ] `RetrieverAgent` parses `ModelCard` instances from raw JSON/markdown text when tool-calling is not triggered.
- [ ] Domain-aware fallback models are returned when search/LLM retrieval returns empty.
- [ ] Official baseline script (`src/baseline/baseline.py`) is detected and evaluated as a candidate.
- [ ] `InitializationPipeline` produces a valid non-empty initial solution $s_0$ with numeric validation score.
- [ ] Full test suite passes 100% green.
