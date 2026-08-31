# Specification: Pydantic AI Built-in WebSearch Capability for RetrieverAgent

## 1. Overview & Strategic Value
Modernize `RetrieverAgent` (`src/problem_2_v2/search/retriever.py`) by leveraging Pydantic AI's native `WebSearch()` capability (`from pydantic_ai.capabilities import WebSearch`). Instead of manually calling a third-party search provider, formatting search result snippets into a prompt string, and passing the concatenated string to the LLM, the `RetrieverAgent` will be equipped with agentic web search capabilities, enabling the LLM to dynamically and autonomously search the web for state-of-the-art models during candidate generation.

## 2. Functional Requirements
- **FR1: Pydantic AI WebSearch Integration**: Equip `RetrieverAgent` with Pydantic AI's `WebSearch()` capability (`capabilities=[WebSearch()]`) to enable dynamic search during agent execution.
- **FR2: Streamlined Prompt Construction**: Simplify prompt formatting so the agent focuses on competition context, objectives, and model card extraction requirements, allowing the LLM to dynamically determine and execute search queries.
- **FR3: Adaptive Hybrid Fallback**: Support both dynamic `WebSearch()` and explicit `SearchProvider` (e.g. `MockSearchProvider`, `DuckDuckGoSearchProvider`, `TavilySearchProvider`) so unit tests and offline environments continue to execute reliably without internet dependencies.
- **FR4: Preserved Contract & Resilience**: Keep `RetrieverAgent.retrieve(spec: TaskSpecification) -> RetrievedCandidates` signature and retain all multi-tier fallbacks (structured output -> text JSON/markdown parsing -> domain-aware starter architectures).

## 3. Non-Functional Requirements & Quality Gates
- **NFR1**: Unit test coverage >80% across modified retriever modules.
- **NFR2**: Strict type safety under `mypy --strict`.
- **NFR3**: Structured observability with Logfire tracing.

## 4. Acceptance Criteria
1. `RetrieverAgent` supports `capabilities=[WebSearch()]` for dynamic retrieval.
2. Unit tests verify dynamic retrieval, text fallback parsing, and mock provider fallback.
3. Zero breaking changes to `InitializationPipeline`, `MLEStarPipeline`, or CLI.
4. Full test suite (`uv run pytest`) passes cleanly.

## 5. Out of Scope
- Altering the `ModelCard` schema or downstream candidate code generation.
