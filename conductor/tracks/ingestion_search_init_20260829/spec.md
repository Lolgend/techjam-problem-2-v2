# Specification: Ingestion & Search-Guided Initialization Phase

## 1. Overview
This track implements the end-to-end task ingestion, web search retrieval, candidate code generation/evaluation, autonomous runtime debugging, and greedy sequential model merging pipeline corresponding to Section 3.1 & Algorithm 1 of the MLE-STAR paper. The outcome is an executable, evaluated, and merged baseline initial solution $s_0$ with validated baseline score $h(s_0)$.

## 2. Functional Requirements

### A. Task Ingestion (`src/problem_2_v2/ingestion/extractor.py`)
- `TaskExtractor`: Pydantic AI Agent that ingests raw markdown problem descriptions and extracts a structured, validated `TaskSpecification`.
- Handles task classification (Tabular, RecSys/Ranking, Vision, NLP, Audio, Multimodal), evaluation metric name & direction (`MAXIMIZE`/`MINIMIZE`), target variable, and dataset directory paths.
- Fallback heuristic parser when working offline or without LLM calls.

### B. Pluggable Search Layer (`src/problem_2_v2/search/`)
- `SearchProvider` (Protocol in `provider.py`): Defines async/sync `search(query: str, num_results: int = 5) -> list[SearchResult]`.
- Implementations:
  - `TavilySearchProvider`: Integrates with Tavily Search API.
  - `GoogleSearchProvider`: Integrates with Google Custom Search JSON API.
  - `DuckDuckGoSearchProvider`: Free web search fallback using duckduckgo.
  - `MockSearchProvider`: Deterministic offline provider for unit and integration testing.
- `RetrieverAgent` ($\mathcal{A}_{\text{retriever}}$ in `retriever.py`): Formulates targeted queries from `TaskSpecification`, queries the search provider, and prompts Pydantic AI agent (Figure 9 prompt) to output $M=4$ structured `ModelCard` instances with concise, runnable code snippets.

### C. Execution Sandbox & Telemetry (`src/problem_2_v2/runner/`)
- `SubprocessRunner` (`sandbox.py`):
  - Creates isolated per-run scratch directories (`runs/<run_id>/sandbox_<candidate_id>/`).
  - Sets up `./input` symlinks / directory mappings so scripts access `./input/<data_files>`.
  - Executes Python scripts via subprocess with configurable per-script timeout (default 600s), memory monitoring, and CPU/CUDA environment variables.
  - Captures stdout/stderr, extracts execution duration, returncode, and parses `Final Validation Performance: {score}` into `ExecutionResult`.
- `DebuggerAgent` ($\mathcal{A}_{\text{debugger}}$ in `debugger.py`):
  - When script execution fails (non-zero returncode, timeout, or missing validation score line), feeds code and traceback into Pydantic AI agent (Figure 19 prompt).
  - Runs repair loop up to `max_debug_rounds` (default 3) before marking execution failed.

### D. Candidate Code Generation & Evaluation (`src/problem_2_v2/initialization/evaluator.py`)
- `CandidateEvaluatorAgent` ($\mathcal{A}_{\text{init}}$):
  - Prompts code generation agent (Figure 10 prompt) with `TaskSpecification` + `ModelCard`.
  - Enforces single-file self-contained code, hold-out validation, 30,000 row subsampling on large datasets, and standardized score printing.
  - Cleans markdown fences, runs through `SubprocessRunner` with `DebuggerAgent` fallback.
  - Evaluates all $M$ candidate models and sorts them into descending performance permutation $\pi$.

### E. Greedy Sequential Model Merging (`src/problem_2_v2/initialization/merger.py`)
- `ModelMergerAgent` ($\mathcal{A}_{\text{merger}}$):
  - Initializes baseline solution $s_0 \leftarrow s_{\text{init}}^{\pi(1)}$ with score $h_{\text{best}} \leftarrow h(s_0)$.
  - For each secondary candidate $i = 2 \dots M$:
    - Prompts merging agent (Figure 11 prompt) to blend candidate $s_{\text{init}}^{\pi(i)}$ into $s_0$ via average ensembling.
    - Executes merged candidate script in sandbox runner.
    - If $h(s_{\text{candidate}}) \ge h_{\text{best}}$: accepts new merged code as $s_0$, updates $h_{\text{best}} \leftarrow h(s_{\text{candidate}})$.
    - If performance degrades or fails: discards candidate and terminates merging loop (Algorithm 1).
- Outputs finalized `PipelineArtifact` containing $s_0$, $h(s_0)$, and lineage history.

## 3. Non-Functional Requirements
- **Determinism:** Explicit random seeds enforced in generated templates and evaluator calls.
- **Modularity:** Every agent is an independent class with typed dependencies, allowing custom mock agents/models during testing.
- **Telemetry & Tracing:** OpenTelemetry / Pydantic Logfire spans for every agent call, search query, code execution, and merge evaluation.
- **Coverage:** >80% test coverage across all new modules and agents.

## 4. Acceptance Criteria
- [ ] `TaskExtractor` extracts accurate `TaskSpecification` from markdown problem files.
- [ ] `RetrieverAgent` queries `SearchProvider` and returns $M$ valid `ModelCard` objects.
- [ ] `SubprocessRunner` executes scripts safely with timeout protection and parses `ExecutionResult`.
- [ ] `DebuggerAgent` successfully repairs erroneous scripts (syntax error, missing import, shape bug) within retry budget.
- [ ] `CandidateEvaluatorAgent` generates executable Python scripts for retrieved models and ranks them by score.
- [ ] `ModelMergerAgent` sequentially merges candidates, retaining improvements and aborting on degradation.
- [ ] Full end-to-end initialization workflow completes and produces initial solution $s_0$ with validated score.
- [ ] All unit and integration tests pass cleanly.
