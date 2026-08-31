# Specification: Retire Task Ingestion Agent & Inject Raw Task Description into Prompts

## Overview
Currently, the pipeline invokes an LLM-based `TaskExtractor` agent during Stage 1 ingestion to parse markdown problem descriptions into structured Pydantic models (`TaskSpecification`), which adds unnecessary LLM latency, cost, and risk of summarizing away critical problem constraints.

This track retires the LLM-based `TaskExtractor` agent and passes the raw markdown task description string directly into downstream agent prompts:
1. **Retriever Agent ($A_{\text{retriever}}$)**: The prompt receives the full raw task description text rather than a reconstructed summary.
2. **Candidate Evaluator Agent ($A_{\text{init}}$)**: The prompt receives the full raw task description text alongside model details and mandatory `evaluate.py` instructions.
3. **Data Usage Checker Agent ($A_{\text{data}}$)**: The prompt receives the full raw task description text to audit dataset file consumption.
4. **Final Artifact Producer / Submission Agent ($A_{\text{finalizer}}$)**: The prompt receives the full raw task description text to ensure full-data training and official submission formatting (`submit.py`).

`TaskSpecification.from_markdown` (deterministic regex/markdown parsing) is retained solely for non-LLM metadata extraction (such as `baseline_score`, `subsample_size`, `metric_direction`, `dataset_files`, `dataset_dir`), completely eliminating the LLM-based task ingestion step from the critical path.

## Functional Requirements
1. **Retire `TaskExtractor` Agent:**
   - Refactor `TaskExtractor` in `src/problem_2_v2/ingestion/extractor.py` to remove the LLM `Agent` and directly return deterministic `TaskSpecification.from_markdown`.
   - Remove LLM extraction calls in `InitializationPipeline.run()` and `MLEStarPipeline.run()`.
2. **Inject Raw Task Description into `RetrieverAgent`:**
   - Update `RetrieverAgent.build_prompt(spec)` so the competition description section directly embeds `spec.raw_description`.
3. **Inject Raw Task Description into `CandidateEvaluatorAgent`:**
   - Update `CandidateEvaluatorAgent.build_prompt(spec, card)` so `{task_description}` in `_EVALUATOR_PROMPT_TEMPLATE` embeds the full raw task description (`spec.raw_description`) and mandates `from evaluate import evaluate` with `(val_user_ids, val_labels, val_predictions)`.
4. **Inject Raw Task Description into `DataUsageCheckerAgent`:**
   - Update `DataUsageCheckerAgent.audit(spec, code)` so the prompt injects `spec.raw_description`.
5. **Inject Raw Task Description into `FinalArtifactProducer`:**
   - Update `FinalArtifactProducer.build_prompt(code, spec)` so the `# Task Description` section contains `spec.raw_description`.
6. **Preserve `TaskSpecification` Metadata Contract:**
   - Add `raw_description: str = Field(default="", description="Raw task markdown description.")` populated during `TaskSpecification.from_markdown(md_text, dataset_dir)`.
   - All deterministic properties (`baseline_score`, `metric_direction`, `subsample_size`, `dataset_files`, `dataset_dir`) remain available for sandbox setup and scoring.

## Acceptance Criteria
- [ ] No LLM agent is invoked for task ingestion during `InitializationPipeline` or `MLEStarPipeline`.
- [ ] `RetrieverAgent`, `CandidateEvaluatorAgent`, `DataUsageCheckerAgent`, and `FinalArtifactProducer` prompts contain the full raw task description text.
- [ ] Candidate evaluator prompt template mandates `from evaluate import evaluate` and documents `evaluate(user_ids, labels, scores)`.
- [ ] All unit and integration tests pass (`pytest`).
- [ ] Ingestion is fast and deterministic, eliminating LLM latency from startup.

## Out of Scope
- Modifying the inner refinement loop agents ($A_{\text{coder}}$, $A_{\text{planner}}$, $A_{\text{extractor}}$) or ensembling planner, which operate on code blocks and solutions rather than task descriptions.
