# Implementation Plan: Targeted Code Block Refinement Phase

## Phase 1: Ablation Generation, Summarization & Code Block Extraction [checkpoint: 513415c]
- [x] Task: Write failing tests for AblationAgent and AblationSummarizerAgent (5f85645)
    - [x] Create `tests/refinement/test_ablation.py` testing Figure 12 ablation script generation and Figure 13 raw log summarization into `AblationReport`
- [x] Task: Implement `AblationAgent` and `AblationSummarizerAgent` in `src/problem_2_v2/refinement/ablation.py` (513415c)
    - [x] Implement Pydantic AI agent generating ablation scripts disabling 2–3 components
    - [x] Implement summarizer agent digesting raw execution outputs into `AblationReport`
    - [x] Verify ablation tests pass
- [x] Task: Write failing tests for CodeBlockExtractorAgent (5f85645)
    - [x] Create `tests/refinement/test_extractor.py` testing Figure 14 target code block extraction and initial plan $p_0$ generation
- [x] Task: Implement `CodeBlockExtractorAgent` in `src/problem_2_v2/refinement/extractor.py` (513415c)
    - [x] Implement Pydantic AI extractor agent returning `TargetCodeBlock` and `RefinementPlan`
    - [x] Implement history context formatting for previously refined blocks
    - [x] Verify extractor tests pass
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Guardrails Layer (Data Leakage & Data Usage Checkers) [checkpoint: 9a6a977]
- [x] Task: Write failing tests for DataLeakageCheckerAgent (74823b0)
    - [x] Create `tests/guardrails/test_leakage.py` testing preprocessing block extraction, leakage detection (Figure 20), and code repair (Figure 21)
- [x] Task: Implement `DataLeakageCheckerAgent` in `src/problem_2_v2/guardrails/leakage.py` (9a6a977)
    - [x] Implement detection and correction agents with AST-safe code replacement
    - [x] Verify leakage tests pass
- [x] Task: Write failing tests for DataUsageCheckerAgent (74823b0)
    - [x] Create `tests/guardrails/test_usage.py` testing missing data source detection and code improvement (Figure 22)
- [x] Task: Implement `DataUsageCheckerAgent` in `src/problem_2_v2/guardrails/usage.py` (9a6a977)
    - [x] Implement usage checking agent and fallback validation
    - [x] Verify usage tests pass
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 3: Targeted Coder & Adaptive Refinement Planner [checkpoint: 4021950]
- [x] Task: Write failing tests for CoderAgent (9087b41)
    - [x] Create `tests/refinement/test_coder.py` testing code block transformation (Figure 15) and AST syntax validation
- [x] Task: Implement `CoderAgent` in `src/problem_2_v2/refinement/coder.py` (4021950)
    - [x] Implement Pydantic AI coder agent returning revised markdown code block
    - [x] Verify coder tests pass
- [x] Task: Write failing tests for RefinementPlannerAgent (9087b41)
    - [x] Create `tests/refinement/test_planner.py` testing history-conditioned adaptive plan generation (Figure 16)
- [x] Task: Implement `RefinementPlannerAgent` in `src/problem_2_v2/refinement/planner.py` (4021950)
    - [x] Implement planner agent formatting attempt trajectory and scores into prompt
    - [x] Verify planner tests pass
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 4: Nested Refinement Pipeline & End-to-End Orchestrator [checkpoint: 7e72a32]
- [x] Task: Write failing tests for RefinementPipeline (2c9b44f)
    - [x] Create `tests/refinement/test_pipeline.py` testing nested $T \times K$ loops, score tracking, best candidate promotion, and JSONL iteration logging
- [x] Task: Implement `RefinementPipeline` in `src/problem_2_v2/refinement/pipeline.py` (7e72a32)
    - [x] Implement Algorithm 2 orchestrator wiring Ablation -> Extractor -> Planner -> Coder -> Guardrails -> Sandbox Runner -> Score Evaluation
    - [x] Implement structured iteration log streaming (`runs/<run_id>/iteration_logs.jsonl`)
    - [x] Verify refinement pipeline tests pass
- [x] Task: Write failing end-to-end integration test (2c9b44f)
    - [x] Create `tests/refinement/test_e2e_refinement.py` testing initialization pipeline output feeding into refinement pipeline
    - [x] Verify end-to-end test passes
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)
