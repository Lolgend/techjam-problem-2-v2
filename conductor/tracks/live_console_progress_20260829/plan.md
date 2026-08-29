# Implementation Plan: Live Real-Time Console Progress, Startup Banner & Stage Telemetry Streaming

## Phase 1: CLI Startup Banner & Master Orchestrator Stage Telemetry
- [ ] Task: Write failing tests for CLI startup banner, summary box, and orchestrator stage announcements
    - [ ] Create `tests/test_live_progress.py` testing banner rendering, stage boundary output, and summary box
- [ ] Task: Implement startup banner and summary box in `src/problem_2_v2/cli.py`
    - [ ] Add ASCII banner with task, dataset, model, and search config details
    - [ ] Add final summary box with duration, baseline score, final validation score, delta, and artifact file paths
- [ ] Task: Implement stage boundary logging in `src/problem_2_v2/orchestrator.py`
    - [ ] Add clear stage announcements: Ingestion -> Parallel Branches -> Ensembling -> Finalization
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 2: Sub-Pipeline Live Telemetry & Score Streaming
- [ ] Task: Write tests for real-time progress emissions across sub-pipelines
    - [ ] Add test cases verifying live console emissions in `tests/test_live_progress.py`
- [ ] Task: Implement live logging in `parallel.py` and `initialization/pipeline.py`
    - [ ] Add branch start/completion status in `ParallelSolutionGenerator`
    - [ ] Add search retrieval, candidate evaluation 1..M scores, and merger s0 score in `InitializationPipeline`
- [ ] Task: Implement live logging in `refinement/pipeline.py`, `ensembling/pipeline.py`, and `execution/finalizer.py`
    - [ ] Add outer loop ablation reports and inner loop refinement plans/scores in `RefinementPipeline`
    - [ ] Add ensembling round strategies and merged scores in `EnsemblePipeline`
    - [ ] Add subsampling removal and full-data training status in `FinalArtifactProducer`
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)

## Phase 3: Full System Integration & Regression Verification
- [ ] Task: Execute full test suite and verify 100% pass rate
    - [ ] Run `uv run pytest --tb=short -q` across all 310+ tests
- [ ] Task: Verify dry-run output formatting on KuaiRand-Pure.md
    - [ ] Run `uv run problem-2-v2 run --task KuaiRand-Pure.md --data src/KuaiRand-Pure-dataset/data --dry-run`
- [ ] Task: Phase Verification & Checkpoint (Refer to workflow.md)
