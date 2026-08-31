# Implementation Plan: Tool-Based Autonomous Search in RetrieverAgent

## Phase 1: Red Phase - Unit Tests for Search Tool
- [ ] Task: Write tests in `tests/search/test_retriever.py` validating that `RetrieverAgent` registers a `search_web` tool and invokes it during multi-turn LLM agent execution
- [ ] Task: Run tests to confirm failure against current static implementation
- [ ] Task: Phase 1 Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Green Phase - Implement Tool-Based RetrieverAgent
- [ ] Task: Centralize the retriever prompt template in `src/problem_2_v2/search/retriever.py`
- [ ] Task: Implement `search_web` function tool on `RetrieverAgent.agent` and `text_agent` calling `self.provider.search(query)` with graceful error catching
- [ ] Task: Refactor `retrieve()` to pass clean prompt to agent and let agent autonomously call `search_web`
- [ ] Task: Run unit tests to confirm all retriever tests pass
- [ ] Task: Phase 2 Verification & Checkpoint (Refer to workflow.md)

## Phase 3: Integration & System Verification
- [ ] Task: Verify integration with `InitializationPipeline` and master pipeline across mock and live providers
- [ ] Task: Run complete test suite (`uv run python -m pytest`) and check code coverage
- [ ] Task: Update `conductor/tech-stack.md` and `conductor/product.md`
- [ ] Task: Phase 3 Verification & Checkpoint (Refer to workflow.md)
