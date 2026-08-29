# Specification: Master Orchestrator, CLI Interface, and Package API Unification

## 1. Overview
This track implements the top-level `MLEStarPipeline` master coordinator, the command-line interface `problem-2-v2`, and unified package `__init__.py` exports across all submodules. This provides the final operational glue that connects problem markdown ingestion directly to production artifact output with zero manual intervention.

## 2. Functional Requirements

### A. Master Orchestrator & Configuration (`src/problem_2_v2/orchestrator.py`, `src/problem_2_v2/config.py`)
- `MLEStarConfig`: Pydantic model configuring all hyperparameters:
  - `model`: str = "openai:gpt-4o"
  - `search_provider`: str = "duckduckgo" ("tavily", "google", "mock")
  - `num_candidates`: int = 4 ($M$)
  - `num_branches`: int = 2 ($L$)
  - `outer_loops`: int = 3 ($T$)
  - `inner_loops`: int = 3 ($K$)
  - `ensemble_rounds`: int = 3 ($R$)
  - `seeds`: list[int] | None = None (default `[42, 123]`)
  - `subsample_size`: int = 30000
  - `timeout_seconds`: int = 600
  - `production_timeout_seconds`: int = 3600
  - `max_debug_rounds`: int = 3
  - `runs_dir`: str = "runs"
  - `final_output_dir`: str = "final"
- `MLEStarPipeline`:
  - Main entry method `run(task_md_path, dataset_dir, run_id=None) -> MLEStarResult`
  - Asynchronous implementation `run_async(task_md_path, dataset_dir, run_id=None) -> MLEStarResult`
  - Coordinates the 5 sequential stages:
    1. **Task Ingestion**: Reads markdown description, parses into `TaskSpecification`.
    2. **Parallel Branches**: Concurrently executes Initialization + Refinement across $L$ seeds to generate candidate artifacts $\{s_1, \dots, s_L\}$.
    3. **Adaptive Ensembling**: Explores blending/stacking strategies over $R$ rounds to produce $s^*_{\text{ens}}$.
    4. **Final Artifact Production**: Strips subsampling, trains on full dataset, serializes models, writes `./final/metrics.json` and `./final/submission.csv`.
    5. **Baseline Comparison**: Computes score delta $\Delta = \text{score}_{\text{final}} - \text{score}_{\text{baseline}}$.
- `MLEStarResult`:
  - Structured output containing `task_spec`, `branch_artifacts`, `ensemble_result`, `final_artifact`, `baseline_score`, `final_score`, `score_delta`, `duration_seconds`, and `success`.

### B. Command-Line Interface (`src/problem_2_v2/cli.py`)
- CLI command `problem-2-v2 run`:
  - `--task`, `-t`: Path to problem markdown file.
  - `--data`, `-d`: Path to dataset directory.
  - `--output`, `-o`: Output directory (default `./final`).
  - `--model`, `-m`: LLM model identifier.
  - `--search-provider`, `-s`: Search backend (`duckduckgo`, `tavily`, `google`, `mock`).
  - `--branches`, `-b`: Number of parallel branches $L$.
  - `--outer-loops`, `-T`: Outer refinement iterations.
  - `--inner-loops`, `-K`: Inner refinement iterations.
  - `--ensemble-rounds`, `-R`: Ensembling rounds.
  - `--seeds`: Comma-separated random seeds.
  - `--dry-run`: Validates task and dataset parsing without executing LLM code generation.
- CLI command `problem-2-v2 version`: Displays version and system info.

### C. Package API Unification (`__init__.py` files)
- Create explicit `__init__.py` files with `__all__` exports for all 7 submodules:
  - `problem_2_v2.ingestion`
  - `problem_2_v2.search`
  - `problem_2_v2.initialization`
  - `problem_2_v2.refinement`
  - `problem_2_v2.guardrails`
  - `problem_2_v2.runner`
  - `problem_2_v2.ensembling`
- Update `problem_2_v2/__init__.py` to export `MLEStarPipeline`, `MLEStarConfig`, `MLEStarResult`, and `main`.

## 3. Non-Functional Requirements
- **Backward Compatibility:** All existing 274 tests must continue to pass without regression.
- **Coverage:** >85% unit and integration test coverage on new orchestrator and CLI modules.
- **Fault Tolerance:** Clear error diagnostics when invalid paths or parameters are supplied.

## 4. Acceptance Criteria
- [ ] `MLEStarConfig` defines all hyperparameter defaults and validates types.
- [ ] `MLEStarPipeline` executes the complete 5-stage workflow end-to-end.
- [ ] `MLEStarResult` records full lineage, final artifacts, and baseline delta.
- [ ] CLI `problem-2-v2 run` handles arguments, runs dry-run check, and launches pipeline.
- [ ] All 7 subpackages contain clean `__init__.py` with `__all__` exports.
- [ ] Full test suite passes 100% green with new unit and CLI integration tests.
