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
