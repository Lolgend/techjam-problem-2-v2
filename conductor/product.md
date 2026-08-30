# Product Definition: MLE-STAR (Machine Learning Engineering Agent)

## Vision & Overview
**MLE-STAR** is a general-purpose, autonomous Machine Learning Engineering agent framework implementing the *MLE-STAR* methodology (Nam et al., 2025). It is designed to autonomously tackle diverse machine learning challenges—from competitive benchmarks (e.g., MLE-bench, Kaggle) to complex domain-specific tasks such as Recommender Systems (e.g., KuaiRand CTR and Ranking), Computer Vision, NLP, and Multimodal problems.

Given a problem description in Markdown format (specifying task objectives, dataset files, baseline scores, evaluation metrics, and constraints) alongside the input data directory, MLE-STAR operates end-to-end with **zero required human intervention**:
1. Reproduces or establishes the baseline pipeline and validation metric.
2. Explores web-retrieved state-of-the-art architectures and candidate models.
3. Conducts ablation-guided targeted code block extraction and nested inner-loop refinement across the entire pipeline (features, models, hyperparameters, loss functions).
4. Explores novel LLM-driven ensembling strategies across parallel candidate solutions.
5. Systematically drives the validation score above the baseline while maintaining comprehensive iteration logs, resource accounting (LLM tokens, GPU hours), and execution safeguards.

## Target Modalities & Problem Scope
- **Recommender Systems & Ranking:** CTR prediction, multi-task ranking, NDCG@K, Recall@K, sequence-aware recommendations (e.g., KuaiRand-Pure, KuaiRand-1k, KuaiRand-27k).
- **Tabular:** Classification, regression, ranking, and time-series forecasting.
- **Computer Vision:** Image classification, image regression, image-to-image (denoising, restoration), and object detection.
- **Natural Language Processing:** Text classification, sequence-to-sequence generation, and text normalization.
- **Audio & Multimodal:** Audio classification, spectrogram modeling, and multimodal fusion.

## Core Capabilities & Architecture

### 1. Baseline Ingestion & Initial Solution Generation
- **Problem Reader & Parser:** Ingests Markdown problem descriptions (`problem.md`), extracting task metadata, target metrics ($h$), dataset schemas, and official baseline benchmarks.
- **Retriever Agent ($A_{\text{retriever}}$):** Queries search engines (Google Search API, Tavily, DuckDuckGo, or local mock for offline testing) with the task description to retrieve $M$ candidate state-of-the-art models and concise example code snippets formatted as structured JSON model cards. Retrieval is resilient: structured output, raw JSON/markdown text parsing, and domain-aware fallback architectures guarantee a non-empty candidate list.
- **Candidate Evaluation Agent ($A_{\text{init}}$):** Implements self-contained executable Python scripts for each candidate model and measures validation performance ($h(s)$) on hold-out validation sets (with automated 30k sample downsampling for fast experimentation). The official baseline starter script (`src/baseline/baseline.py`) is seeded as the first candidate when present, so $s_0$ can match or exceed the baseline score. Validation metrics are computed with the official baseline harness (`src/baseline/evaluate.py`; for KuaiRand-Pure, `primary = (GAUC + nDCG@5) / 2.0`), which sandboxed scripts import directly via the injected `PYTHONPATH`, guaranteeing scores strictly match the competition metric.
- **Merging Agent ($A_{\text{merger}}$):** Sequentially integrates non-dominated candidate models into an initial consolidated baseline ($s_0$) using an averaging ensemble as long as validation score improves. If all merge attempts fail, the best individual candidate is preserved rather than returning an empty solution.

### 2. Targeted Code Block Extraction & Refinement (Nested Exploration)
- **Outer Loop ($T$ iterations):**
  - **Ablation Study Agent ($A_{\text{abl}}$):** Generates ablation scripts ($a_t$) isolating 2–3 specific pipeline components (e.g., feature transformations, interaction terms, scaling, imputers, model backbones, loss functions).
  - **Summarization Module ($A_{\text{summarize}}$):** Extracts clean component-impact summaries from raw execution outputs.
  - **Extractor Module ($A_{\text{extractor}}$):** Identifies the single code block ($c_t$) with highest performance impact, avoiding previously modified blocks for exploration diversity, and drafts initial improvement plan ($p_0$).
- **Inner Loop ($K$ iterations):**
  - **Coder Agent ($A_{\text{coder}}$):** Rewrites target code block $c_t$ into refined block $c_t^k$ according to the plan.
  - **Replacement & Evaluation:** Replaces $c_t$ with $c_t^k$ in the full script and records validation score and delta over baseline ($\Delta(m) = \text{score}_{\text{agent}} - \text{score}_{\text{baseline}}$).
  - **Planner Agent ($A_{\text{planner}}$):** Proposes subsequent novel refinement plans ($p_k$) conditioned on historical attempt trajectory and scores.
  - **Best Candidate Selection:** Promotes improved solutions to $s_{t+1}$ only upon score gain.

### 3. LLM-Driven Ensembling Strategy Exploration
- Evaluates $L$ candidate solutions produced from parallel runs.
- **Ensemble Planner Agent ($A_{\text{ens\_planner}}$):** Proposes $R$ rounds of novel ensembling techniques (e.g., weighted averaging with grid search, stacking with meta-learners, rank averaging, out-of-fold blending).
- **Ensembler Agent ($A_{\text{ensembler}}$):** Implements the ensemble plan into a single executable script producing `./final/submission.csv`.
- Selects optimal ensemble $s_{\text{ens}}^*$ based on validation metrics.

### 4. Convergence & Resource Accounting
- **Convergence Detection:** Automatically determines convergence when the validation score fails to improve by more than $\epsilon$ over $N$ consecutive iterations or when the compute/token budget is exhausted.
- **Telemetry & Resource Tracking:** Tracks cumulative token consumption (input + output tokens) and GPU runtime (GPU-hours / seconds) per iteration and across the entire run.
- **Iteration Run-Logger:** Generates structured per-iteration run logs capturing:
  - Iteration number & targeted pipeline component
  - Hypothesis & strategic intent
  - Applied code diff
  - Resulting validation metric & delta over baseline
  - Error/recovery events and manual intervention count (target: 0)

### 5. Safeguards, Verification & Execution Runtime
- **Execution Guardrail Pipeline:** A unified orchestrator (`ExecutionGuardrailPipeline`) sequencing the data leakage check, data usage check, sandbox execution, and automatic debugging loop, configured via `ExecutionConfig` (timeouts, retry rounds, guardrail toggles). Both the refinement and ensembling pipelines route all script execution through it.
- **Local Subprocess Runner:** Isolated execution with configurable timeouts, GPU/CUDA acceleration, memory monitoring, and robust exception trapping.
- **Debugging Agent ($A_{\text{debugger}}$):** Iteratively corrects execution errors and runtime tracebacks up to max retry rounds without human intervention.
- **Data Leakage Checker ($A_{\text{leakage}}$):** Extracts preprocessing code blocks, inspects for test/val contamination during train steps, and auto-corrects leaky logic.
- **Data Usage Checker ($A_{\text{data}}$):** Cross-references task description with dataset files to ensure all auxiliary features/modalities are consumed.
- **Final Artifact Producer ($A_{\text{finalizer}}$):** Restores full training data (removes temporary subsampling), trains on the complete dataset, and produces the production-ready `./final/` output (serialized model files, `metrics.json` evaluation scores, and `submission.csv`). Submissions are written in the exact official schema (`row_id,user_id,video_id,score` per `submit.py`) following the deterministic `data.load()` test row order, and are automatically verified with `submit.py --check` after being copied to the output directory.

## User Interface & Integration
- **Master Orchestrator:** `MLEStarPipeline` coordinates the 5-stage workflow (task ingestion, parallel branches, adaptive ensembling, final artifact production, baseline comparison) from a single `run()` / `run_async()` entry point, configured via `MLEStarConfig`.
- **CLI Interface:** `problem-2-v2 run --task <problem.md> --data <dir> --output <dir> [--model ... --search-provider ... --branches ... --dry-run]` and `problem-2-v2 version`; `--dry-run` validates inputs without executing code generation. Running renders an immediate startup banner (task/type/metric/baseline/dataset/model/search/loops), streams live stage and score telemetry (branches, candidates, refinement plans, ensemble rounds, finalization) with `flush=True`, and closes with a final summary box of duration, scores, delta, artifact paths, and automated submission verification status (`submit.py --check` PASSED/FAILED/not applicable).
- **Python Library API:** Programmatic invocation of individual subagents or the full pipeline via `MLEStarPipeline`, returning a structured `MLEStarResult` (lineage, artifacts, baseline delta).
- **Observability:** Integrated with Pydantic Logfire for complete span-level tracing of LLM reasoning, code modifications, score trajectories, and tool executions.
