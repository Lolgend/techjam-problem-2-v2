# Implementation Plan: Adaptive Ensembling Phase

## Phase 1: Parallel Candidate Solution Generation [checkpoint: 14f874e]
- [x] Task: Write failing tests for ParallelSolutionGenerator (b5f518d)
    - [x] Create `tests/ensembling/test_parallel.py` testing concurrent execution across $L$ seeds and isolated sandbox management
- [x] Task: Implement `ParallelSolutionGenerator` in `src/problem_2_v2/ensembling/parallel.py` (14f874e)
    - [x] Implement async coordinator running Initialization + Refinement per seed
    - [x] Return list of $L$ validated `PipelineArtifact` instances
    - [x] Verify parallel generation tests pass
- [x] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Adaptive Ensemble Planner ($\mathcal{A}_{\text{ens\_planner}}$)
- [ ] Task: Write failing tests for EnsemblePlannerAgent
    - [ ] Create `tests/ensembling/test_planner.py` testing initial plan $e_0$ generation and history-conditioned novel plans $e_r$ (Figure 17 prompt)
- [ ] Task: Implement `EnsemblePlannerAgent` in `src/problem_2_v2/ensembling/planner.py`
    - [ ] Implement Pydantic AI planner agent generating initial and subsequent ensemble strategies
    - [ ] Format candidate solutions and attempt trajectories into prompt context
    - [ ] Verify planner tests pass
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 3: Code Ensembler ($\mathcal{A}_{\text{ensembler}}$)
- [ ] Task: Write failing tests for EnsemblerAgent
    - [ ] Create `tests/ensembling/test_ensembler.py` testing single-file merged script synthesis (Figure 18 prompt), `./final/submission.csv` creation, and AST validation
- [ ] Task: Implement `EnsemblerAgent` in `src/problem_2_v2/ensembling/ensembler.py`
    - [ ] Implement Pydantic AI ensembler agent generating unified executable Python code
    - [ ] Clean markdown fences, validate AST syntax, and execute in `SubprocessRunner` with `DebuggerAgent` fallback
    - [ ] Verify ensembler tests pass
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 4: Iterative Ensemble Pipeline & End-to-End Orchestrator
- [ ] Task: Write failing tests for EnsemblePipeline
    - [ ] Create `tests/ensembling/test_pipeline.py` testing $R$ optimization rounds, score delta tracking, winner selection ($s^*_{\text{ens}}$), and JSONL iteration logging
- [ ] Task: Implement `EnsemblePipeline` in `src/problem_2_v2/ensembling/pipeline.py`
    - [ ] Implement Algorithm 3 iterative loop coordinating Planner -> Ensembler -> Sandbox Runner -> Score Evaluation
    - [ ] Select optimal solution $s^*_{\text{ens}} = \arg\max h(s)$ with fallback to best individual candidate
    - [ ] Stream structured iteration logs (`runs/<run_id>/iteration_logs.jsonl`) and generate final `PipelineArtifact`
    - [ ] Verify ensemble pipeline tests pass
- [ ] Task: Write failing end-to-end integration test
    - [ ] Create `tests/ensembling/test_e2e_ensembling.py` testing full workflow from $L$ candidates through $R$ ensembling rounds to final submission
    - [ ] Verify end-to-end test passes
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)
