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

## Phase 2: Guardrails Layer (Data Leakage & Data Usage Checkers)
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
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 3: Targeted Coder & Adaptive Refinement Planner
- [ ] Task: Write failing tests for CoderAgent
    - [ ] Create `tests/refinement/test_coder.py` testing code block transformation (Figure 15) and AST syntax validation
- [ ] Task: Implement `CoderAgent` in `src/problem_2_v2/refinement/coder.py`
    - [ ] Implement Pydantic AI coder agent returning revised markdown code block
    - [ ] Verify coder tests pass
- [ ] Task: Write failing tests for RefinementPlannerAgent
    - [ ] Create `tests/refinement/test_planner.py` testing history-conditioned adaptive plan generation (Figure 16)
- [ ] Task: Implement `RefinementPlannerAgent` in `src/problem_2_v2/refinement/planner.py`
    - [ ] Implement planner agent formatting attempt trajectory and scores into prompt
    - [ ] Verify planner tests pass
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 4: Nested Refinement Pipeline & End-to-End Orchestrator
- [ ] Task: Write failing tests for RefinementPipeline
    - [ ] Create `tests/refinement/test_pipeline.py` testing nested $T \times K$ loops, score tracking, best candidate promotion, and JSONL iteration logging
- [ ] Task: Implement `RefinementPipeline` in `src/problem_2_v2/refinement/pipeline.py`
    - [ ] Implement Algorithm 2 orchestrator wiring Ablation -> Extractor -> Planner -> Coder -> Guardrails -> Sandbox Runner -> Score Evaluation
    - [ ] Implement structured iteration log streaming (`runs/<run_id>/iteration_logs.jsonl`)
    - [ ] Verify refinement pipeline tests pass
- [ ] Task: Write failing end-to-end integration test
    - [ ] Create `tests/refinement/test_e2e_refinement.py` testing initialization pipeline output feeding into refinement pipeline
    - [ ] Verify end-to-end test passes
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)
