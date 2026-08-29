# Product Definition: MLE-STAR (Machine Learning Engineering Agent)

## Vision & Overview
**MLE-STAR** is an autonomous Machine Learning Engineering agent framework implementing the *MLE-STAR* methodology (Nam et al., 2025). It consumes a machine learning problem description in Markdown format (containing problem specifications, task type, data descriptions, evaluation metrics, and constraints) alongside the raw dataset directory, and fully automates the end-to-end ML solution development pipeline.

MLE-STAR moves beyond naive whole-script generation by combining external web search for state-of-the-art model retrieval, ablation-guided targeted code block extraction, nested inner-loop refinement, novel LLM-driven ensembling, and strict safeguards against bugs, data leakage, and unused dataset features.

## Target Modalities & Problem Scope
- **Tabular:** Classification, regression, ranking, and time-series.
- **Computer Vision:** Image classification, image regression, image-to-image (e.g., denoising), and object detection.
- **Natural Language Processing:** Text classification, sequence-to-sequence generation, and text normalization.
- **Audio:** Audio classification, event detection, and spectrogram-based modeling.
- **Multimodal:** Mixed tabular, image, and text inputs.

## Core Capabilities & Architecture
1. **Initial Solution Generation via Web Search:**
   - **Retriever Agent ($A_{\text{retriever}}$):** Queries search engines (Google Search, Tavily, DuckDuckGo, or local mock for offline testing) with the task description to retrieve $M$ candidate state-of-the-art models and concise example code snippets formatted as structured JSON model cards.
   - **Candidate Evaluation Agent ($A_{\text{init}}$):** Implements self-contained executable Python scripts for each candidate model and measures validation performance ($h(s)$) on hold-out validation sets (with automated 30k sample downsampling for fast experimentation).
   - **Merging Agent ($A_{\text{merger}}$):** Sequentially integrates non-dominated candidate models into an initial consolidated baseline ($s_0$) using an averaging ensemble as long as validation score improves.

2. **Targeted Code Block Extraction & Refinement (Nested Exploration):**
   - **Outer Loop ($T$ iterations):**
     - **Ablation Study Agent ($A_{\text{abl}}$):** Generates ablation scripts ($a_t$) isolating 2–3 specific pipeline components (e.g., categorical encoding, scaling, imputers, model backbones).
     - **Summarization Module ($A_{\text{summarize}}$):** Extracts clean component-impact summaries from raw execution outputs.
     - **Extractor Module ($A_{\text{extractor}}$):** Identifies the single code block ($c_t$) with highest performance impact, avoiding previously modified blocks for exploration diversity, and drafts initial improvement plan ($p_0$).
   - **Inner Loop ($K$ iterations):**
     - **Coder Agent ($A_{\text{coder}}$):** Rewrites target code block $c_t$ into refined block $c_t^k$ according to the plan.
     - **Replacement & Evaluation:** Replaces $c_t$ with $c_t^k$ in the full script and records validation score.
     - **Planner Agent ($A_{\text{planner}}$):** Proposes subsequent novel refinement plans ($p_k$) conditioned on historical attempt trajectory and scores.
     - **Best Candidate Selection:** Promotes improved solutions to $s_{t+1}$ only upon score gain.

3. **LLM-Driven Ensembling Strategy Exploration:**
   - Evaluates $L$ candidate solutions produced from parallel runs.
   - **Ensemble Planner Agent ($A_{\text{ens\_planner}}$):** Proposes $R$ rounds of novel ensembling techniques (e.g., weighted averaging with grid search, stacking with meta-learners, rank averaging, out-of-fold blending).
   - **Ensembler Agent ($A_{\text{ensembler}}$):** Implements the ensemble plan into a single executable script producing `./final/submission.csv`.
   - Selects optimal ensemble $s_{\text{ens}}^*$ based on validation metrics.

4. **Safeguards, Verification & Execution Runtime:**
   - **Local Subprocess Runner:** Isolated execution with configurable timeouts, GPU/CUDA acceleration, memory monitoring, and robust exception trapping.
   - **Debugging Agent ($A_{\text{debugger}}$):** Iteratively corrects execution errors and runtime tracebacks up to max retry rounds.
   - **Data Leakage Checker ($A_{\text{leakage}}$):** Extracts preprocessing code blocks, inspects for test/val contamination during train steps, and auto-corrects leaky logic.
   - **Data Usage Checker ($A_{\text{data}}$):** Cross-references task description with dataset files to ensure all auxiliary features/modalities are consumed.
   - **Test & Submission Generator ($A_{\text{test}}$):** Restores full training data (removes temporary subsampling) and generates formatted `./final/submission.csv`.

## User Interface & Integration
- **CLI Interface:** `problem-2-v2 run --task <problem.md> --input <dir> --output <dir>` with real-time progress logging.
- **Python Library API:** Programmatic invocation and orchestration of individual subagents or end-to-end pipeline runs.
- **Observability:** Integrated with Pydantic Logfire for complete span-level tracing of LLM reasoning, code modifications, score trajectories, and tool executions.
