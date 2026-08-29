# Specification: Targeted Code Block Refinement Phase

## 1. Overview
This track implements the core nested search and targeted code block refinement engine of MLE-STAR corresponding to Section 3.2, Section 3.4, and Algorithm 2 of the paper. It consists of an outer exploration loop ($T$ iterations) targeting specific ML components via ablation studies and an inner refinement loop ($K$ iterations) optimizing the targeted code block using historical feedback, guardrails, and adaptive LLM planning.

## 2. Functional Requirements

### A. Ablation Generation & Summarization (`src/problem_2_v2/refinement/ablation.py`)
- `AblationAgent` ($\mathcal{A}_{\text{abl}}$): Generates an ablation test script from active solution $s_t$ with previous ablation summaries $\{\mathcal{T}_{\text{abl}}^i\}_{i=0}^{t-1}$ as context (Figure 12 prompt), modifying/disabling 2–3 distinct components without touching test data.
- `AblationSummarizerAgent` ($\mathcal{A}_{\text{summarize}}$): Executes the ablation script via `SubprocessRunner`, parses stdout/logs, and generates a structured `AblationReport` identifying the highest-impact pipeline component (Figure 13 prompt).

### B. Targeted Code Block Extraction (`src/problem_2_v2/refinement/extractor.py`)
- `CodeBlockExtractorAgent` ($\mathcal{A}_{\text{extractor}}$): Analyzes current solution $s_t$, ablation summary, and previously modified blocks $\{c_i\}_{i=0}^{t-1}$ to extract the exact code snippet $c_t$ with highest impact and generate the initial refinement plan $p_0$ (Figure 14 prompt).

### C. Guardrail Checking & Repair (`src/problem_2_v2/guardrails/`)
- `DataLeakageCheckerAgent` ($\mathcal{A}_{\text{leakage}}$ in `leakage.py`): Inspects data preprocessing code blocks (Figure 20 prompt). If leakage is found, generates a corrected code block (Figure 21 prompt) and replaces it prior to evaluation.
- `DataUsageCheckerAgent` ($\mathcal{A}_{\text{data}}$ in `usage.py`): Cross-references solution code with `TaskSpecification` dataset metadata (Figure 22 prompt) to ensure all supplied dataset files and modalities are utilized.

### D. Code Block Refinement & Adaptive Planning (`src/problem_2_v2/refinement/`)
- `CoderAgent` ($\mathcal{A}_{\text{coder}}$ in `coder.py`): Transforms extracted target block $c_t$ according to active plan $p_k$ (Figure 15 prompt), outputting only the revised code block $c_t^k$.
- `RefinementPlannerAgent` ($\mathcal{A}_{\text{planner}}$ in `planner.py`): Proposes subsequent novel refinement plans $p_k$ for $k=1 \dots K-1$ conditioned on history of attempted plans and validation scores (Figure 16 prompt).
- Code Patching & Validation: Direct string replacement $s_t^k = s_t.\text{replace}(c_t, c_t^k)$ with AST syntax check and fallback whitespace normalization.

### E. Nested Optimization Orchestrator (`src/problem_2_v2/refinement/pipeline.py`)
- `RefinementPipeline`:
  - Executes Outer Loop ($t = 0 \dots T-1$):
    - Runs $\mathcal{A}_{\text{abl}} \rightarrow \mathcal{A}_{\text{summarize}} \rightarrow \mathcal{A}_{\text{extractor}}$ to obtain $c_t$ and $p_0$.
    - Executes Inner Loop ($k = 0 \dots K-1$):
      - If $k > 0$: calls $\mathcal{A}_{\text{planner}}$ to generate $p_k$.
      - Calls $\mathcal{A}_{\text{coder}}$ to generate $c_t^k$.
      - Patches script, passes guardrails (`DataLeakageChecker` & `DataUsageChecker`).
      - Runs sandbox evaluation (with `DebuggerAgent` retry loop).
      - Records validation score and updates best candidate if score improves.
      - Streams structured iteration log record to `runs/<run_id>/iteration_logs.jsonl`.
    - Promotes best inner candidate to $s_{t+1}$ if score exceeds $s_t$.
  - Returns finalized `PipelineArtifact` containing the best solution script and complete lineage trajectory.

## 3. Non-Functional Requirements
- **Determinism:** Explicit random seeds enforced in generated templates and score evaluation.
- **Robustness:** Handles missing imports, syntax errors, and runtime crashes via `DebuggerAgent`.
- **Telemetry:** Spans and metric logging integrated with Pydantic Logfire.
- **Test Coverage:** >80% unit and integration test coverage across all new modules.

## 4. Acceptance Criteria
- [ ] `AblationAgent` generates executable ablation scripts disabling 2–3 components.
- [ ] `AblationSummarizerAgent` accurately parses raw execution logs into `AblationReport`.
- [ ] `CodeBlockExtractorAgent` extracts exact substring $c_t$ and valid initial plan $p_0$.
- [ ] `CoderAgent` refines target code block without removing subsampling.
- [ ] `DataLeakageCheckerAgent` detects and fixes data leakage in preprocessing pipelines.
- [ ] `DataUsageCheckerAgent` verifies full dataset consumption.
- [ ] `RefinementPlannerAgent` produces non-repetitive plans conditioned on attempt history.
- [ ] `RefinementPipeline` executes full nested $T \times K$ loops and returns improved solution script.
- [ ] All unit and integration tests pass cleanly.
