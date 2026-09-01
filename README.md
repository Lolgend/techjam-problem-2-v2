# Modified MLE-STAR (baseline seeding): Autonomous Machine Learning Research Agent for Recommender Systems

**MLE-STAR** is a state-of-the-art, fully autonomous Machine Learning Engineering agent tailored for competitive recommender system challenges. Built with [Pydantic AI](https://ai.pydantic.dev/) and [Logfire](https://logfire.pydantic.dev/), MLE-STAR automates the complete research-and-development lifecycle: problem ingestion, web search retrieval of domain-specific architectures, isolated sandbox evaluations, empirical component ablation, targeted inner/outer code refinement, multi-model ensembling, and 100% full-dataset production retraining.

---

## 📋 Table of Contents

- [Pipeline Execution by Stages](#-pipeline-execution-by-stages)
- [Specialized AI Agent Roster](#-specialized-ai-agent-roster)
- [Safety Guardrails & Self-Healing](#-safety-guardrails--self-healing)
- [Installation & Setup](#-installation--setup)
- [Reflections & Limitations](#limitations--reflections)

---

## ⚡ Pipeline Execution by Stages

### **Stage 1: Task Ingestion & Extraction**
* Parses the problem description markdown ([`KuaiRand-Pure.md`](./KuaiRand-Pure.md)) into a structured, validated `TaskSpecification`.
* Connects the Logfire OpenTelemetry exporter (`send_to_logfire='if-token-present'`) and starts unbuffered terminal streaming.

### **Stage 2: Parallel Model Initialization (Algorithm 1, $L=2$ Branches)**
* **Multi-Branch Diversity:** Concurrently spawns Branch 0 (seed 0) and Branch 1 (seed 1).
* **Web Search Retrieval:** Queries the web for recent top-performing recommender models (DIN, DeepFM, BST, LightGBM LambdaRank, Two-Tower) and auto-seeds the official competition baseline starter script.
* **Sandbox Evaluation:** Trains each candidate independently in its own subprocess sandbox on a fast 30,000-row sample.
* **Sequential Model Merging:** Evaluates combining the top 2 candidate model families into a unified baseline $s_0$.

### **Stage 3: Targeted Refinement Loop (Algorithm 2, $T=3$ Outer, $K=3$ Inner)**
* **Outer Loop ($T=3$ Ablation Cycles):** Systematically mutes components (e.g. Loss function, ID embeddings, sequential attention layers) to empirically discover which module represents the performance bottleneck.
* **Inner Loop ($K=3$ Targeted Mutations):**
  * `RefinementPlannerAgent` formulates an ML hypothesis (e.g., *Switch Pointwise BCE Loss to Pairwise BPR Ranking Loss*).
  * `CoderAgent` writes the localized code mutation.
  * `DataLeakageCheckerAgent` & `DataUsageCheckerAgent` verify safety.
  * The sandbox evaluates the new code. If $\Delta > 0$, the mutation is accepted and committed to the lineage.

### **Stage 4: Adaptive Ensembling (Algorithm 3, $R=3$ Rounds)**
* Takes the refined winning pipelines from Branch 0 and Branch 1.
* Iteratively explores **Rank-Weighted Probability Averaging** and **LightGBM Meta-Stacking** over 3 rounds to achieve maximum generalization.

### **Stage 5: Production Finalization**
* Automatically removes the 30,000 subsample cap.
* Retrains the winning pipeline on **100% of the full dataset**.
* Computes final baseline delta $\Delta(m) = \text{Final Score} - 0.6016$ and exports standard competition submission files to `./final/`.

---

## 🤖 Specialized AI Agent Roster

MLE-STAR coordinates a network of 14 specialized, domain-focused LLM agents built with [Pydantic AI](https://ai.pydantic.dev/):

| Agent Name | Symbol | Source Module | Input / Prompt Schema | Output Schema | Primary Function |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **Task Extractor** | $A_{\text{extractor}}$ | [`ingestion/extractor.py`](./src/problem_2_v2/ingestion/extractor.py) | Markdown text + dataset path | [`TaskSpecification`](./src/problem_2_v2/contracts/task.py) | Parses problem description into structured task specification. |
| **Web Retriever** | $A_{\text{retriever}}$ | [`search/retriever.py`](./src/problem_2_v2/search/retriever.py) | Search query + snippet context | `list[`[`ModelCard`](./src/problem_2_v2/contracts/search.py)`]` | Searches for state-of-the-art architectures and extracts model cards. |
| **Candidate Evaluator** | $A_{\text{init}}$ | [`initialization/evaluator.py`](./src/problem_2_v2/initialization/evaluator.py) | `TaskSpecification` + `ModelCard` | Python script (`str`) | Writes self-contained Python scripts for initial candidate models. |
| **Sequential Merger** | $A_{\text{merger}}$ | [`initialization/merger.py`](./src/problem_2_v2/initialization/merger.py) | Base solution + Reference solution | Merged Python script (`str`) | Generates hybrid architectures combining top-ranked models. |
| **Ablation Agent** | $A_{\text{abl}}$ | [`refinement/ablation.py`](./src/problem_2_v2/refinement/ablation.py) | Solution script + Ablation history | Standalone ablation script (`str`) | Generates ablated scripts disabling 2-3 parts of the training pipeline. |
| **Ablation Summarizer** | $A_{\text{summarize}}$ | [`refinement/ablation.py`](./src/problem_2_v2/refinement/ablation.py) | Ablation code + Process stdout/stderr | [`AblationReport`](./src/problem_2_v2/contracts/refinement.py) | Executes ablation and ranks components by optimization headroom. |
| **Component Extractor** | $A_{\text{block}}$ | [`refinement/extractor.py`](./src/problem_2_v2/refinement/extractor.py) | Solution code + `AblationReport` | [`TargetCodeBlock`](./src/problem_2_v2/contracts/refinement.py) + [`RefinementPlan`](./src/problem_2_v2/contracts/refinement.py) | Isolates highest-headroom code block and drafts initial plan $p_0$. |
| **Refinement Planner** | $A_{\text{planner}}$ | [`refinement/planner.py`](./src/problem_2_v2/refinement/planner.py) | `TargetCodeBlock` + Attempt history | [`RefinementPlan`](./src/problem_2_v2/contracts/refinement.py) | Formulates targeted hypotheses conditioned on previous trials. |
| **Coder Agent** | $A_{\text{coder}}$ | [`refinement/coder.py`](./src/problem_2_v2/refinement/coder.py) | `TargetCodeBlock` + `RefinementPlan` | Refined code block (`str`) | Writes precise Python code mutations for the targeted block. |
| **Debugger Agent** | $A_{\text{debugger}}$ | [`runner/debugger.py`](./src/problem_2_v2/runner/debugger.py) | Broken code + Exception traceback | [`DebugOutcome`](./src/problem_2_v2/runner/debugger.py) (Repaired script) | Analyzes tracebacks (`stderr`) and repairs runtime/syntax bugs. |
| **Data Leakage Checker** | $A_{\text{leakage}}$ | [`guardrails/leakage.py`](./src/problem_2_v2/guardrails/leakage.py) | Full solution Python script | [`DataLeakageStatus`](./src/problem_2_v2/contracts/guardrails.py) + Repaired code | Static audit detecting test label leakage or invalid split mixing. |
| **Data Usage Checker** | $A_{\text{data}}$ | [`guardrails/usage.py`](./src/problem_2_v2/guardrails/usage.py) | Solution script + `TaskSpecification` | [`DataUsageStatus`](./src/problem_2_v2/contracts/guardrails.py) + Improved code | Verifies that all required dataset tables and features are ingested. |
| **Ensemble Planner** | $A_{\text{ens\_planner}}$ | [`ensembling/planner.py`](./src/problem_2_v2/ensembling/planner.py) | Branch artifacts + Attempt history | [`EnsembleStrategy`](./src/problem_2_v2/contracts/guardrails.py) | Selects optimal ensembling strategy (averaging vs stacking). |
| **Ensembler Agent** | $A_{\text{ensembler}}$ | [`ensembling/ensembler.py`](./src/problem_2_v2/ensembling/ensembler.py) | Candidate artifacts + `EnsembleStrategy` | Full ensemble script (`str`) | Synthesizes a unified single-file Python ensemble script. |
| **Finalizer Agent** | $A_{\text{finalizer}}$ | [`execution/finalizer.py`](./src/problem_2_v2/execution/finalizer.py) | Winning solution + `TaskSpecification` | [`FinalArtifact`](./src/problem_2_v2/execution/finalizer.py) | Strips subsampling, retrains on 100% data, and exports deliverables. |

---

## 🛡 Safety Guardrails & Self-Healing

1. **Isolated Subprocess Sandboxes ([`src/problem_2_v2/runner/sandbox.py`](./src/problem_2_v2/runner/sandbox.py)):**
   * Each candidate executes in an isolated scratch directory (`runs/<run_id>/branch_0/sandbox_cand*/`) with automatic dataset mounting to `./input/`.
   * Hard wall-clock timeout (default 600s) kills hanging loops or memory exhaustion.
2. **Autonomous Debugger Self-Healing ([`src/problem_2_v2/runner/debugger.py`](./src/problem_2_v2/runner/debugger.py)):**
   * When a training script raises an exception (`returncode != 0`), the Debugger Agent captures `stderr`, analyzes the traceback, revises the code, and re-executes up to 3 repair rounds.
3. **Data Leakage Guardrail ([`src/problem_2_v2/guardrails/leakage.py`](./src/problem_2_v2/guardrails/leakage.py)):**
   * Verifies that temporal splits are strictly respected and test labels are never accessed during training or validation.

---

## 📦 Installation & Setup

### Prerequisites
* Python 3.10 or higher
* [uv](https://github.com/astral-sh/uv) package manager

```bash
# Clone repository
git clone https://github.com/Lolgend/techjam-problem-2-v2.git
cd "Problem 2 V2"

# Install all dependencies into virtual environment
uv sync
```

---

## 🚀 CLI Usage & Options

Reproduce my results (may differ):

```bash
uv run problem-2-v2 run --task KuaiRand-Pure.md --data src/KuaiRand-Pure-dataset/data --output ./final --model "deepseek:deepseek-v4-pro" -k "[DEEPSEEK API KEY]" -v --logfire-token "[PYDANTIC LOGIFRE KEY]" -b 1 -K 4 -T 4 -R 0 --search-provider tavily --search-api-key "[TAVILY API KEY]" --max-tokens 38400
```

---

## Limitations & Reflections

Currently the agent still tends to hallucinate sometimes and not follow contraints/instructions. I primarily only tested it on DeepSeek so I'm not sure if other LLM models will produce better results. Weirdly, there are issues with the LLM hitting max output tokens only on deepseek v4 flash and not pro, maybe a prompting issue. Issues are primarily LLM/Agent based and may be able to be fixed with better prompting techniques. The whole thing runs end to end properly.

If given more time, I would explore different prompts or prompting techniques to have the model adhere to instructions better. I would also look at the full suite of agents and experiment on reordering or restructuring them if possible. Also with more time, I'd let it run beyond the 6 hour cap to see what it can achieve. I would also like to do some optimizations in terms of the baseline seeding and the data file access. Currently each sandbox has it's own copy of files which is not very space optimized. Given more time, I'd explore alternative solutions.



