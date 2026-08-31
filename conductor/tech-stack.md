# Technology Stack: MLE-STAR

## Core Runtime & Packaging
- **Language:** Python >= 3.10
- **Package Manager & Virtualenv:** `uv` (fast dependency resolution and lockfile management)
- **Configuration & Data Modeling:** `pydantic` >= 2.13.5

## Agent & Orchestration Layer
- **Agent Framework:** `pydantic-ai` (Typed agents, tool calling, structured outputs, validation)
- **Supported LLM Providers:**
  - Google Gemini (`gemini-2.5-pro`, `gemini-2.0-flash`)
  - Anthropic Claude (`claude-3-7-sonnet`, `claude-3-5-sonnet`)
  - OpenAI (`gpt-4o`, `o1`, `o3-mini`)
- **Web Search Tools:** Modular retriever layer supporting Google Search API (CSE), Tavily API, DuckDuckGo, and Mock/Offline file-based search, built on `httpx`.

## Machine Learning & Data Processing Ecosystem
- **Core Scientific:** `numpy`, `pandas`, `scipy`
- **Classic & Boosted ML:** `scikit-learn`, `lightgbm`, `xgboost`, `catboost`
- **Deep Learning & Vision/NLP:** `torch` (CUDA enabled), `torchvision`, `timm`, `transformers`, `accelerate`

## Execution Engine & Runtime Safeguards
- **Subprocess Runner:** `subprocess` with isolated working directories, stream piping, execution timeout control, and CUDA environment passthrough.
- **Data Leakage & Preprocessing Verification:** AST static analysis + Pydantic AI LLM checker.

## Observability, Telemetry & UI
- **Tracing & Telemetry:** `logfire` (OpenTelemetry instrumentation, LLM traces, span hierarchies, run metrics)
- **CLI & Formatting:** `typer`, `rich` (live tables, score progression, colored logs)

## Quality & Development Tooling
- **Testing:** `pytest`, `pytest-asyncio`, `pytest-cov` (coverage target: >80%, spec NFR: >90%)
- **Linter & Formatter:** `ruff`
- **Type Checking:** `mypy --strict` (with `pydantic.mypy` plugin)

> **2026-08-29:** Added `pytest-cov` and the strict mypy `pydantic.mypy` plugin
> configuration while implementing the Pydantic data contracts track
> (`core_data_contracts_20260829`).
>
> **2026-08-29:** Added `pydantic-ai`, `ddgs` (DuckDuckGo search;
> successor to `duckduckgo-search`), `httpx`, and
> `logfire` as runtime dependencies while implementing the ingestion &
> search-guided initialization track (`ingestion_search_init_20260829`).
> Peak memory monitoring is deferred: the sandbox currently provides
> isolation, timeouts, and score-line parsing only.
>
> **2026-08-29:** Unified the execution safeguards into an
> `ExecutionGuardrailPipeline` orchestrator and added the
> `FinalArtifactProducer` finalizer while implementing the execution
> guardrail modules track (`execution_guardrails_20260829`). No new runtime
> dependencies. Peak memory monitoring remains deferred (the spec's "memory
> limits" requirement is covered by sandbox isolation, timeouts, and
> `./input` mapping).
>
> **2026-08-29:** Added the `MLEStarPipeline` master orchestrator and the
> `problem-2-v2` CLI (stdlib `argparse`) while implementing the master
> orchestrator track (`master_orchestrator_cli_20260829`). No new runtime
> dependencies. The `typer`/`rich` CLI tooling listed above is not used;
> live console progress is provided by a stdlib `console` helper
> (`announce`/`format_score`/`format_delta`, `flush=True`) added in the
> live console progress track (`live_console_progress_20260829`); rich
> interactive tables remain future work.
>
> **2026-08-30:** Fixed Windows Proactor `WinError 10038` socket teardown
> crashes by configuring `asyncio.WindowsSelectorEventLoopPolicy` on
> Windows (`configure_event_loop_policy` in the orchestrator, invoked by
> the CLI and `MLEStarPipeline.run`). `DuckDuckGoSearchProvider` is now
> thread-safe (serialized via `threading.Lock`, fresh scoped
> `DDGS(timeout=20)` sessions per call, graceful `[]` fallback on
> socket/network/rate-limit errors) while fixing the
> `windows_asyncio_socket_fix_20260829` track.
>
> **2026-08-30:** Hardened candidate retrieval and initial solution
> generation while implementing the robust candidate retrieval track
> (`robust_candidate_retrieval_20260830`): `RetrieverAgent` falls back from
> structured output to raw JSON/markdown parsing to domain-aware starter
> cards (never empty), `InitializationPipeline` can seed the official
> baseline script as candidate 1, and `ModelMergerAgent` preserves the best
> individual on failed merges. No new runtime dependencies.
>
> **2026-08-30:** Fixed duplicate relative-path resolution in
> `SubprocessRunner` while fixing the sandbox path resolution track
> (`sandbox_path_resolution_fix_20260830`): `prepare_sandbox` and
> `run_code` now resolve sandbox and `solution.py` paths to absolute via
> `.resolve()`, eliminating `[Errno 2]` when `runs_dir` is relative.
>
> **2026-08-31:** Hardened the data leakage guardrail and execution pipeline
> (`leakage_guardrail_fix_20260831`): `DataLeakageCheckerAgent` now utilizes a
> multi-tier patching strategy (exact match → fuzzy matching via
> `find_matching_block` with indentation alignment → full-script rewrite
> fallback). `ExecutionGuardrailPipeline` now runs a check→repair→re-check retry
> loop up to `max_leakage_retries` (default 5), supports strict abortion mode via
> `strict_leakage: bool = False` raising `LeakageEnforcementError`, and emits
> unambiguous Logfire events (`execution.leakage_repaired`,
> `execution.leakage_unrepaired`, `guardrails.leakage_repair.succeeded`).
>
> **2026-08-31:** Refactored `RetrieverAgent` to provide web search as an
> autonomous callable function tool (`search_web`) registered directly on the
> Pydantic AI `Agent` (`retriever_tool_websearch_20260831`). Centralized the
> retriever prompt template and system instructions in `retriever.py`, allowing the
> LLM to formulate search queries and retrieve evidence dynamically across all model
> providers (OpenAI, DeepSeek, Gemini, Anthropic) without static prompt injection.
>
> **2026-08-31:** Retired the LLM-based task ingestion agent in favor of fast,
> deterministic markdown parsing preserving `raw_description`
> (`task_ingestion_passthrough_20260831`). Injected the full raw task description
> string directly into the prompts for `RetrieverAgent`, `CandidateEvaluatorAgent`,
> `DataUsageCheckerAgent`, and `FinalArtifactProducer`, and explicitly mandated the
> official `evaluate.py` harness (`evaluate(user_ids, labels, scores, k=5)`) in
> candidate generation prompts.

