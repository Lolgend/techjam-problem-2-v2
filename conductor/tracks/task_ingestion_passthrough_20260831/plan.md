# Plan: Retire Task Ingestion Agent & Inject Raw Task Description into Prompts

## Phase 1: Ingestion Architecture & `TaskSpecification` Contract Update [checkpoint: e61949f]
- [x] Task: Write failing unit tests for `TaskSpecification.raw_description` and non-LLM `TaskExtractor` (e61949f)
  - [x] Add tests in `tests/contracts/test_task.py` asserting `TaskSpecification.from_markdown` preserves `raw_description`.
  - [x] Add tests in `tests/ingestion/test_extractor.py` verifying `TaskExtractor` operates deterministically without LLM calls.
- [x] Task: Update `TaskSpecification` and `TaskExtractor` implementation (e61949f)
  - [x] Add `raw_description: str = Field(default="", ...)` to `TaskSpecification` in `src/problem_2_v2/contracts/task.py`.
  - [x] Update `from_markdown` to store `md_text.strip()` as `raw_description`.
  - [x] Refactor `TaskExtractor` in `src/problem_2_v2/ingestion/extractor.py` to remove the LLM `Agent` and directly return `TaskSpecification.from_markdown`.
  - [x] Run `pytest tests/contracts/test_task.py tests/ingestion/test_extractor.py` to confirm green phase.
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md) (e61949f)

## Phase 2: Prompt Injections across Target Agents & Evaluate Harness Mandate
- [ ] Task: Write failing unit tests for prompt generation with raw task description
  - [ ] Update `tests/search/test_retriever.py` to verify `RetrieverAgent.build_prompt` embeds `raw_description`.
  - [ ] Update `tests/initialization/test_evaluator.py` to verify `CandidateEvaluatorAgent.build_prompt` embeds `raw_description` and mandatory `evaluate.py` harness instructions.
  - [ ] Update `tests/guardrails/test_usage.py` to verify `DataUsageCheckerAgent` prompt embeds `raw_description`.
  - [ ] Update `tests/execution/test_finalizer.py` to verify `FinalArtifactProducer.build_prompt` embeds `raw_description`.
- [ ] Task: Update Agent Implementations & Prompts
  - [ ] Update `RetrieverAgent.build_prompt` in `src/problem_2_v2/search/retriever.py`.
  - [ ] Update `CandidateEvaluatorAgent.build_prompt` in `src/problem_2_v2/initialization/evaluator.py` with `raw_description` and mandatory `evaluate.py` instructions.
  - [ ] Update `DataUsageCheckerAgent.audit` in `src/problem_2_v2/guardrails/usage.py`.
  - [ ] Update `FinalArtifactProducer.build_prompt` in `src/problem_2_v2/execution/finalizer.py`.
  - [ ] Run `pytest` on the updated agent test files to confirm green phase.
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 3: Pipeline & Orchestrator Integration & Full Verification
- [ ] Task: Update `InitializationPipeline` and `MLEStarPipeline`
  - [ ] Update `InitializationPipeline.run()` in `src/problem_2_v2/initialization/pipeline.py` to bypass any LLM ingestion.
  - [ ] Align `MLEStarConfig.num_candidates` in `src/problem_2_v2/config.py` with test suite defaults (4).
  - [ ] Update `tests/test_orchestrator.py` and `tests/initialization/test_pipeline.py`.
- [ ] Task: Run Full Test Suite & Coverage Check
  - [ ] Run `uv run pytest` across all 466+ tests.
  - [ ] Run `uv run ruff check src` and `uv run mypy src`.
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)
