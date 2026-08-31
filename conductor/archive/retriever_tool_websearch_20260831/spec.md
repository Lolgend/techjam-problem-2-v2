# Specification: Tool-Based Autonomous Search in RetrieverAgent

## Overview
Refactor `RetrieverAgent` in `src/problem_2_v2/search/retriever.py` so that web search is provided as a dynamic callable tool to the Pydantic AI `Agent` (`search_web(query: str) -> str`) instead of pre-executing a fixed search query prior to prompting the model. The agent will formulate its own queries, call the search tool autonomously as needed, and distill candidate `ModelCard` architectures.

## Functional Requirements
1. **Dynamic Search Tool:**
   - Define a `search_web(query: str, num_results: int = 5) -> str` tool registered directly on `RetrieverAgent.agent` and `text_agent`.
   - Tool calls `self.provider.search(query, num_results=num_results)`.
   - Catches provider runtime errors (e.g. network timeouts, rate-limits) and returns informative feedback so the agent can adapt or proceed without raising unhandled exceptions.
2. **Centralized & Customizable Prompting:**
   - Centralize the prompt template and system instructions in `src/problem_2_v2/search/retriever.py` in a single location for easy maintenance.
   - Use clean markdown structure:
     ```markdown
     # Competition
     {task_name / task_type}
     {description}
     Evaluation metric: {metric_name} ({metric_direction})

     # Your task
     - List {num_candidates} recent effective models and their example codes to win the above competition.

     # Requirement
     - The example code should be concise and simple.
     - You must provide an example code, i.e., do not just mention GitHubs or papers.
     ```
3. **Universal Model Compatibility:**
   - Standard function tool calling works across all model backends (`OpenAIChatModel`, DeepSeek, Gemini, Anthropic, FunctionModel, TestModel) without requiring OpenAI Responses API or specific search grounding support.
4. **Fallback & Robustness:**
   - Preserve structured output with regex/json fallback extraction and domain-aware default model cards if LLM extraction fails.

## Acceptance Criteria
- Unit tests verify agent tool-calling during retrieval with `FunctionModel` and `MockSearchProvider`.
- All existing tests pass.
- Prompt template is modifiable in a single place in `retriever.py`.
