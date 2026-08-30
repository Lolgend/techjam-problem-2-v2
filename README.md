# MLE-STAR: Autonomous Machine Learning Research Agent for Recommender Systems

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/managed%20by-uv-261230.svg)](https://github.com/astral-sh/uv)
[![Pydantic Logfire](https://img.shields.io/badge/observability-Logfire-ff4387.svg)](https://logfire.pydantic.dev)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**MLE-STAR** is a state-of-the-art, fully autonomous Machine Learning Engineering agent tailored for competitive recommender system challenges. Built with [Pydantic AI](https://ai.pydantic.dev/) and [Logfire](https://logfire.pydantic.dev/), MLE-STAR automates the complete research-and-development lifecycle: problem ingestion, web search retrieval of domain-specific architectures, isolated sandbox evaluations, empirical component ablation, targeted inner/outer code refinement, multi-model ensembling, and 100% full-dataset production retraining.

---

## 📋 Table of Contents

- [Competition Benchmark: KuaiRand-Pure](#-competition-benchmark-kuairand-pure)
- [End-to-End System Architecture](#-end-to-end-system-architecture)
- [Pipeline Execution by Stages](#-pipeline-execution-by-stages)
- [Specialized AI Agent Roster](#-specialized-ai-agent-roster)
- [Inter-Agent Interaction Dynamics & Sequence Flows](#-inter-agent-interaction-dynamics--sequence-flows)
- [Safety Guardrails & Self-Healing](#-safety-guardrails--self-healing)
- [Installation & Setup](#-installation--setup)
- [CLI Usage & Options](#-cli-usage--options)
- [Deliverables & Output Schema](#-deliverables--output-schema)
- [Live Telemetry & Observability](#-live-telemetry--observability)
- [Automated Test Suite](#-automated-test-suite)

---

## 🎯 Competition Benchmark: KuaiRand-Pure

MLE-STAR is specifically tuned to maximize ranking performance on the **KuaiRand-Pure** micro-video recommender benchmark:

| Dimension | Specification |
| :--- | :--- |
| **Domain** | Sequential short-video recommendation feed (27k users × 7.6k videos) |
| **Prediction Target** | `long_view` (Binary engagement label: 1 if watched past threshold, 0 otherwise) |
| **Primary Metric** | $\text{Primary Score} = \text{mean}(\text{GAUC}, \text{nDCG@5})$ (Within-User Group AUC + nDCG at rank 5) |
| **Official Baseline** | **`0.6016`** (Factorization Machine starter pipeline) |
| **Data Partitioning** | • **Train:** `log_standard_4_08_to_4_21_pure.csv`<br>• **Validation:** `log_standard_4_22_to_5_08_pure.csv` (First 50%)<br>• **Test:** `log_standard_4_22_to_5_08_pure.csv` (Last 50%) |
| **Auxiliary Signals** | `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `play_time_ms` |

---

## 🏗 End-to-End System Architecture

```mermaid
flowchart TD
    CLI["CLI Command Ingestion\nproblem-2-v2 run --task KuaiRand-Pure.md"] --> STAGE1["Stage 1: Task Ingestion & Telemetry Setup\n• Parses KuaiRand-Pure.md (Target: long_view, Metric: GAUC+nDCG@5, Baseline: 0.6016)\n• Mounts Logfire cloud tracing & Live Console Telemetry"]

    STAGE1 --> STAGE2["Stage 2: Launch 2 Parallel Seed Branches (L=2)\n(Branch 0: seed=0 | Branch 1: seed=1)"]

    subgraph Branch ["Per-Branch Pipeline Execution (Concurrent)"]
        STAGE2 --> RETRIEVE["1. Web Search & Model Retrieval (A_retriever)\n• DuckDuckGo queries -> Retrieves 4 SOTA Recommender Cards\n• Seeds Official Factorization Machine Baseline as Candidate 1"]
        
        RETRIEVE --> CAND_EVAL["2. Candidate Sandbox Evaluation (A_init)\n• Evaluates 5 Candidates on 30k sample in isolated sandboxes\n• (e.g. Baseline: 0.4386, LightGBM LambdaRank: 0.7412, DIN: ...)\n• DebuggerAgent auto-fixes any syntax/runtime crashes"]
        
        CAND_EVAL --> MERGE["3. Sequential Model Merging (A_merger)\n• Takes top candidate and attempts to merge #2 candidate features\n• Verifies if merged variant beats initial best score -> Produces s0"]

        MERGE --> REFINE["Stage 3: Targeted Refinement Loop (T=3 Outer, K=3 Inner)"]
        
        subgraph RefinementLoop ["Algorithm 2: Outer & Inner Loops"]
            REFINE --> ABLATION["Outer Loop (T=1..3):\n• AblationAgent (A_abl): Tests removing Loss, Features, Architecture\n• ExtractorAgent: Isolates lowest-performing high-headroom code block"]
            ABLATION --> INNER["Inner Loop (K=1..3):\n• PlannerAgent: Formulates targeted hypothesis\n• CoderAgent: Mutates Python code\n• Guardrails (A_leakage, A_data): Validates safety\n• Sandbox: Trains & evaluates new validation score\n• If Score > Best: Accept mutation & record diff"]
            INNER --> ABLATION
        end
    end

    Branch --> STAGE4["Stage 4: Adaptive Ensembling (Algorithm 3, R=3 Rounds)\n• Gathers best refined solutions from Branch 0 and Branch 1\n• EnsemblePlanner chooses optimal blending strategy (Rank-weighted averaging / GBDT stacking)\n• Trains joint ensemble and validates score gain"]

    STAGE4 --> STAGE5["Stage 5: Production Finalization (A_finalizer)\n• Automatically removes 30k subsample cap\n• Retrains winning architecture on 100% full dataset in production sandbox\n• Generates Final Deliverables:\n  ├── ./final/submission.csv (row_id, user_id, video_id, score)\n  ├── ./final/metrics.json (Validation Score, Baseline, Delta Δ)\n  └── ./final/model.* (Serialized model weights)"]
```

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

## 🔄 Inter-Agent Interaction Dynamics & Sequence Flows

### 1. Refinement Loop & Guardrail Feedback Sequence (Algorithm 2)

During each outer cycle $t \in [1..T]$, the system identifies the most promising component via empirical ablation, extracts the code block, and conducts $K$ inner mutation iterations with safety guardrails and self-healing debugging:

```mermaid
sequenceDiagram
    autonumber
    participant Orchestrator as RefinementPipeline
    participant AblAgent as AblationAgent (A_abl)
    participant AblSum as AblationSummarizer (A_summarize)
    participant Extractor as CodeBlockExtractor (A_block)
    participant Planner as RefinementPlanner (A_planner)
    participant Coder as CoderAgent (A_coder)
    participant Leakage as DataLeakageChecker (A_leakage)
    participant Usage as DataUsageChecker (A_data)
    participant Sandbox as SubprocessRunner (Sandbox)
    participant Debugger as DebuggerAgent (A_debugger)
    participant Logger as CentralIterationLogger

    Note over Orchestrator,Logger: Outer Exploration Loop (T=3): Identify Headroom
    Orchestrator->>AblAgent: generate_ablation(current_code, history)
    AblAgent-->>Orchestrator: ablation_code
    Orchestrator->>AblSum: summarize(ablation_code, run_id)
    AblSum->>Sandbox: run_code(ablation_code)
    Sandbox-->>AblSum: ExecutionResult (stdout/stderr)
    AblSum-->>Orchestrator: AblationReport (highest_impact_component)
    Orchestrator->>Extractor: extract(solution, AblationReport, previous_blocks)
    Extractor-->>Orchestrator: TargetCodeBlock + initial RefinementPlan (p0)

    Note over Orchestrator,Logger: Inner Optimization Loop (K=3): Targeted Mutations
    loop Inner Iteration (k = 1..K)
        alt k == 0
            Note over Orchestrator: Use initial plan p0 from Extractor
        else k > 0
            Orchestrator->>Planner: next_plan(TargetCodeBlock, previous_attempts)
            Planner-->>Orchestrator: RefinementPlan (pk)
        end
        Orchestrator->>Coder: refine(TargetCodeBlock, RefinementPlan)
        Coder-->>Orchestrator: refined_block_code
        Note over Orchestrator: patch_script(current_code, target_block, refined_block)
        
        Note over Orchestrator,Usage: ExecutionGuardrailPipeline Pass
        Orchestrator->>Leakage: audit(patched_code)
        alt Data Leakage Detected
            Leakage-->>Orchestrator: DataLeakageStatus(is_leaking=True) + auto_repaired_code
        else Clean
            Leakage-->>Orchestrator: DataLeakageStatus(is_leaking=False)
        end

        Orchestrator->>Usage: audit(spec, guarded_code)
        alt Missing Data Sources
            Usage-->>Orchestrator: DataUsageStatus + improved_code
        else All Data Used
            Usage-->>Orchestrator: DataUsageStatus(all_data_used=True)
        end

        Note over Orchestrator,Debugger: Subprocess Execution & Self-Healing
        Orchestrator->>Sandbox: run_code(final_guarded_code)
        Sandbox-->>Orchestrator: ExecutionResult
        opt returncode != 0 or score is None
            loop up to 3 repair rounds
                Orchestrator->>Debugger: debug(broken_code, stderr_traceback)
                Debugger->>Sandbox: run_code(repaired_code)
                Sandbox-->>Debugger: ExecutionResult
            end
            Debugger-->>Orchestrator: DebugOutcome(code, result, recovered)
        end

        Orchestrator->>Logger: append(IterationLogEntry [hypothesis, diff, metrics, errors])
        Note over Orchestrator: Evaluate greedy score improvement: h(s_cand) > h(s_best)
    end
```

### 2. Model Initialization & Sequential Merging Sequence (Algorithm 1)

Before refinement, candidates retrieved from web search and the official baseline starter are ranked and greedily merged into a unified starting point $s_0$:

```mermaid
sequenceDiagram
    autonumber
    participant Pipeline as InitializationPipeline
    participant Retriever as RetrieverAgent
    participant Evaluator as CandidateEvaluator
    participant Merger as ModelMergerAgent
    participant Debugger as DebuggerAgent
    participant Sandbox as SubprocessRunner

    Pipeline->>Retriever: retrieve(spec)
    Retriever-->>Pipeline: ModelCards [c1, c2, c3, c4] + Official Baseline
    Pipeline->>Evaluator: evaluate_all(spec, ModelCards)
    loop Each Candidate Card
        Evaluator->>Sandbox: run_code(candidate_script)
        Sandbox-->>Evaluator: ExecutionResult (Validation Score)
    end
    Evaluator-->>Pipeline: Ranked Candidates [c1 (0.7412), c2 (0.6850), c3 (0.6120)]

    Note over Pipeline,Merger: Greedy Sequential Merging
    Pipeline->>Merger: merge(spec, ranked_candidates)
    Note over Merger: s0 = c1 (top candidate)
    loop k = 2 to M
        Merger->>Merger: Prompt LLM to integrate c[k] into s0
        Merger->>Debugger: debug(merged_script)
        Debugger->>Sandbox: run_code(merged_script)
        Sandbox-->>Debugger: ExecutionResult (merged_score)
        Debugger-->>Merger: DebugOutcome
        alt merged_score >= current_best_score
            Note over Merger: Accept merge, s0 = s_merged, continue
        else merged_score < current_best_score or Error
            Note over Merger: Reject merge, abort greedy loop
        end
    end
    Merger-->>Pipeline: MergeOutcome (Final consolidated s0)
```

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

Run the full autonomous MLE-STAR pipeline with a single command:

```bash
uv run problem-2-v2 run \
  --task KuaiRand-Pure.md \
  --data src/KuaiRand-Pure-dataset/data \
  --output ./final \
  --model "deepseek:deepseek-v4-flash" \
  -k "<your-llm-api-key>" \
  -v
```

### Complete CLI Argument Reference

| Flag | Short | Default | Description |
| :--- | :---: | :---: | :--- |
| `--task` | `-t` | *(Required)* | Path to the task specification markdown (e.g. `KuaiRand-Pure.md`). |
| `--data` | `-d` | *(Required)* | Path to the dataset directory containing CSV tables. |
| `--output` | `-o` | `./final` | Directory where submission and metric deliverables are written. |
| `--model` | `-m` | `openai:gpt-4o` | LLM model identifier (`deepseek:deepseek-v4-flash`, `openai:gpt-4o`, `anthropic:claude-3-5-sonnet`, `gemini-1.5-pro`, etc.). |
| `--api-key` | `-k` | `None` | API key (automatically maps to `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`). |
| `--base-url` | — | `None` | Custom API base URL (e.g. `https://api.deepseek.com` or `https://openrouter.ai/api/v1`). |
| `--search-provider` | `-s` | `duckduckgo` | Search backend (`duckduckgo`, `tavily`, `google`, `mock`). |
| `--branches` | `-b` | `2` | Number of parallel seed exploration branches ($L$). |
| `--outer-loops` | `-T` | `3` | Number of outer component ablation cycles ($T$). |
| `--inner-loops` | `-K` | `3` | Number of inner targeted refinement iterations per cycle ($K$). |
| `--ensemble-rounds` | `-R` | `3` | Number of adaptive ensembling rounds ($R$). |
| `--verbose` | `-v` | `False` | Stream live training stdout/stderr and detailed agent diagnostics. |
| `--logfire-token` | — | `None` | Logfire write token for streaming live traces to the web dashboard. |
| `--dry-run` | — | `False` | Validate task specification and inputs without launching model training. |

---

## 📊 Deliverables & Output Schema

Upon completion, the pipeline outputs competition-ready artifacts into `./final/`:

```text
final/
├── submission.csv      # Test predictions formatted for competition leaderboard
├── metrics.json        # Comprehensive validation score, baseline, and delta summary
└── model.*             # Serialized winning model checkpoint & ensemble weights
```

### 1. `submission.csv` Schema
```csv
row_id,user_id,video_id,score
0,1234,5678,0.8492
1,1234,9012,0.6120
2,5678,3456,0.9145
```

### 2. `metrics.json` Schema
```json
{
  "task_name": "KuaiRand-Pure Recommender Ranking",
  "metric_name": "primary",
  "baseline_score": 0.6016,
  "final_validation_score": 0.7412,
  "delta_improvement": 0.1396,
  "converged": true,
  "total_iterations": 18,
  "total_branches": 2
}
```

### 3. Iteration Run Logs (`runs/<run_id>/iteration_logs.jsonl`)
Every code mutation records its **hypothesis**, **applied unified diff**, **resulting validation score**, and **debugger recovery actions** for complete auditability.

---

## 📡 Live Telemetry & Observability

MLE-STAR features native integration with [Pydantic Logfire](https://logfire.pydantic.dev):

```bash
# In Command Prompt (cmd.exe)
set LOGFIRE_TOKEN=your-logfire-write-token
uv run problem-2-v2 run --task KuaiRand-Pure.md --data src/KuaiRand-Pure-dataset/data -k "your-api-key"

# In PowerShell
$env:LOGFIRE_TOKEN = "your-logfire-write-token"
uv run problem-2-v2 run --task KuaiRand-Pure.md --data src/KuaiRand-Pure-dataset/data -k "your-api-key"
```

* **Live Spans:** Real-time visual timeline across all 44+ stages (Parallel Branches $\rightarrow$ Retrieval $\rightarrow$ Initialization $\rightarrow$ Refinement $\rightarrow$ Ensembling $\rightarrow$ Finalization).
* **LLM Inspector:** Inspect exact prompts, agent thoughts, structured JSON cards, token usage, and latency.

---

## 🧪 Automated Test Suite

MLE-STAR includes a comprehensive test suite covering unit tests, contract validations, integration pipelines, and mock LLM executions:

```bash
# Run all 354 test cases
uv run pytest

# Run with concise summary
uv run pytest --tb=short -q
```

---

## 📜 References & Acknowledgements

* **MLE-Bench:** J. S. Chan et al., *"Evaluating Machine Learning Agents on Machine Learning Engineering,"* OpenAI, 2024.
* **AIDE:** Z. Jiang et al., *"AI-Driven Exploration in the Space of Code,"* 2025.
* **KuaiRand:** Kuaishou Technology, *"KuaiRand: An Unbiased Sequential Recommendation Dataset with Millions of Interactions,"* 2022.
* **Pydantic AI & Logfire:** Built on the robust [Pydantic](https://pydantic.dev/) agent ecosystem.
