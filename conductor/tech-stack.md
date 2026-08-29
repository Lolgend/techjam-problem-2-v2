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
- **Web Search Tools:** Modular retriever layer supporting Google Search API (CSE), Tavily API, DuckDuckGo, and Mock/Offline file-based search.

## Machine Learning & Data Processing Ecosystem
- **Core Scientific:** `numpy`, `pandas`, `scipy`
- **Classic & Boosted ML:** `scikit-learn`, `lightgbm`, `xgboost`, `catboost`
- **Deep Learning & Vision/NLP:** `torch` (CUDA enabled), `torchvision`, `timm`, `transformers`, `accelerate`

## Execution Engine & Runtime Safeguards
- **Subprocess Runner:** `subprocess` with isolated working directories, stream piping, execution timeout control, and memory protection.
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
