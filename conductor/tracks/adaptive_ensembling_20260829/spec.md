# Specification: Adaptive Ensembling Phase

## 1. Overview
This track implements the Adaptive Ensembling Phase of MLE-STAR corresponding to Section 3.3 and Algorithm 3 of the paper. It takes $L$ distinct solution pipelines generated across parallel branches with different seeds, and autonomously discovers optimal model blending and stacking strategies over $R$ iterative rounds using an Ensemble Planner ($\mathcal{A}_{\text{ens\_planner}}$) and Code Ensembler ($\mathcal{A}_{\text{ensembler}}$) to produce the optimal final solution $s^*_{\text{ens}}$.

## 2. Functional Requirements

### A. Parallel Solution Generation (`src/problem_2_v2/ensembling/parallel.py`)
- `ParallelSolutionGenerator`: Runs the full pipeline (Initialization + Refinement) across $L$ parallel branches (default $L=2$) with distinct random seeds.
- Executes concurrently using `asyncio` with isolated scratch sandbox environments for each branch.
- Returns list of $L$ validated `PipelineArtifact` instances $\{s_1, \dots, s_L\}$ with individual baseline scores.

### B. Adaptive Ensemble Planning (`src/problem_2_v2/ensembling/planner.py`)
- `EnsemblePlannerAgent` ($\mathcal{A}_{\text{ens\_planner}}$):
  - Ingests $L$ candidate Python solutions and history of attempted ensemble plans and scores $\{(e_j, h(s_{\text{ens}}^j))\}_{j=0}^{r-1}$.
  - Generates initial plan $e_0$ (e.g. weighted probability averaging or simple averaging).
  - Proposes novel subsequent plans $e_r$ for $r=1 \dots R-1$ (e.g., meta-learner stacking with Logistic Regression/Ridge, rank averaging, out-of-fold blending, dynamic thresholding) using the Figure 17 prompt.

### C. Unified Code Ensembling (`src/problem_2_v2/ensembling/ensembler.py`)
- `EnsemblerAgent` ($\mathcal{A}_{\text{ensembler}}$):
  - Ingests $L$ candidate Python solutions and active ensemble plan $e_r$.
  - Prompts Pydantic AI agent (Figure 18 prompt) to synthesize a single-file, self-contained Python program combining all base pipelines, executing the ensemble plan, computing validation metric on holdout split, and writing `./final/submission.csv`.
  - Cleans markdown fences, validates AST syntax, and executes in `SubprocessRunner` with `DebuggerAgent` fallback.

### D. Iterative Ensemble Optimization Orchestrator (`src/problem_2_v2/ensembling/pipeline.py`)
- `EnsemblePipeline`:
  - Implements Algorithm 3:
    1. Evaluates initial ensemble plan $e_0 \rightarrow s_{\text{ens}}^0 \rightarrow h(s_{\text{ens}}^0)$.
    2. Runs iterative loop for $r = 1 \dots R-1$ with history feedback.
    3. Selects optimal solution $s^*_{\text{ens}} = \arg\max h(s)$ across all $R$ ensemble scripts and $L$ individual candidates (ensuring ensembling never degrades below best single candidate).
    4. Writes structured iteration logs to `runs/<run_id>/iteration_logs.jsonl` and returns final `PipelineArtifact`.

## 3. Non-Functional Requirements
- **Determinism:** Seed controls across PyTorch, NumPy, scikit-learn, LightGBM in all ensemble scripts.
- **Robustness:** Handles syntax/runtime bugs via `DebuggerAgent` without stalling the ensemble search.
- **Observability:** Full span tracing and score tracking via Pydantic Logfire.
- **Coverage:** >80% test coverage across all ensembling modules.

## 4. Acceptance Criteria
- [ ] `ParallelSolutionGenerator` generates $L$ distinct, executable candidate solutions concurrently.
- [ ] `EnsemblePlannerAgent` generates valid initial plan $e_0$ and novel adaptive plans $e_r$ based on history.
- [ ] `EnsemblerAgent` synthesizes single-file executable ensemble scripts that output `./final/submission.csv`.
- [ ] `EnsemblePipeline` executes $R$ iterations, records score trajectory, and selects optimal $s^*_{\text{ens}}$.
- [ ] Unit and integration test suite passes 100% green with >80% code coverage.
