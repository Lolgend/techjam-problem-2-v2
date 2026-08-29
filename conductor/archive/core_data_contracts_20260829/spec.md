# Specification: Core Architecture & Pydantic Data Contracts

## 1. Overview
This track implements the foundational data models and contract layer for the MLE-STAR agent framework using Pydantic V2. It establishes strict, type-safe data schemas for task ingestion, subprocess execution telemetry, web-search candidate retrieval, ablation study reporting, targeted code block extraction, inner-loop refinement planning, ensembling strategies, and guardrail validations (data leakage & data usage).

## 2. Functional Requirements

### A. Task and Environment Models (`src/problem_2_v2/contracts/task.py`)
- `TaskType` (Enum): `TABULAR_CLASSIFICATION`, `TABULAR_REGRESSION`, `RECOMMENDER_RANKING`, `IMAGE_CLASSIFICATION`, `IMAGE_TO_IMAGE`, `TEXT_CLASSIFICATION`, `SEQ_TO_SEQ`, `AUDIO_CLASSIFICATION`, `MULTIMODAL`.
- `MetricDirection` (Enum): `MAXIMIZE` (higher is better, e.g. NDCG@K, Recall@K, Accuracy, AUROC, R2), `MINIMIZE` (lower is better, e.g. RMSE, LogLoss, MAE, RMLSE). Includes helper `is_better(score_a, score_b) -> bool` and `delta(score, baseline) -> float`.
- `TaskSpecification`: Parses markdown problem descriptions (`task_name`, `task_type`, `description`, `metric_name`, `metric_direction`, `target_variable`, `dataset_dir`, `dataset_files`, `baseline_score`, `constraints`, `subsample_size=30000`). Includes markdown parser helper `from_markdown(md_text, dataset_dir)`.
- `ExecutionResult`: Captures subprocess execution outcomes (`success: bool`, `stdout: str`, `stderr: str`, `returncode: int`, `duration_seconds: float`, `validation_score: Optional[float]`, `error_traceback: Optional[str]`, `gpu_memory_mb: Optional[float]`). Includes regex parser for `Final Validation Performance: {score}`.
- `PipelineArtifact`: Lineage tracker containing `version: int`, `full_code: str`, `validation_score: float`, `parent_version: Optional[int]`, `applied_diff: Optional[str]`, `iteration_stage: str`, `timestamp: datetime`.

### B. Initialization & Search Schemas (`src/problem_2_v2/contracts/search.py`)
- `ModelCard`: Structured output model for retrieved candidate models (`model_name: str`, `rationale: str`, `example_code: str`, `library_dependencies: list[str]`). Includes markdown fence extractor to clean raw code.
- `RetrievedCandidates`: Container holding `candidates: list[ModelCard]`, `query_used: str`, `total_found: int`.

### C. Targeted Refinement Schemas (`src/problem_2_v2/contracts/refinement.py`)
- `ComponentCategory` (Enum): `DATA_PREPROCESSING`, `FEATURE_ENGINEERING`, `MODEL_ARCHITECTURE`, `LOSS_AND_OPTIMIZER`, `HYPERPARAMETERS`, `POST_PROCESSING`.
- `AblationVariant`: Captures an isolated ablation experiment (`variant_id: str`, `component_name: str`, `category: ComponentCategory`, `hypothesis: str`, `modified_code_block: str`, `ablation_code: str`).
- `AblationResultItem`: Single ablation outcome (`variant_id: str`, `validation_score: float`, `delta_from_baseline: float`, `summary: str`).
- `AblationReport`: Aggregated report (`baseline_score: float`, `ablation_results: list[AblationResultItem]`, `highest_impact_component: str`, `raw_log_summary: str`).
- `TargetCodeBlock`: Extracted segment for optimization (`raw_code: str`, `category: ComponentCategory`, `start_line: Optional[int]`, `end_line: Optional[int]`, `initial_plan: str`). Includes helper `replace_in(full_script, new_code) -> str` with AST validation.
- `RefinementPlan`: Inner-loop planning model (`plan_id: str`, `natural_language_plan: str`, `target_subcomponents: list[str]`, `expected_gain: str`, `iteration_index: int`).

### D. Ensembling & Guardrail Schemas (`src/problem_2_v2/contracts/guardrails.py`)
- `EnsembleMethod` (Enum): `SIMPLE_AVERAGE`, `WEIGHTED_AVERAGE`, `STACKING_META_LEARNER`, `RANK_AVERAGING`, `BLENDING`.
- `EnsembleStrategy`: Captures ensembling approach (`method: EnsembleMethod`, `natural_language_plan: str`, `meta_learner_type: Optional[str]`, `candidate_solution_ids: list[str]`, `code_template: Optional[str]`).
- `DataLeakageStatus`: Prompt-compatible leakage detector (`leakage_status: str`, `is_leaking: bool`, `suspicious_code_block: Optional[str]`, `corrected_code_block: Optional[str]`, `explanation: str`). Includes validator accepting `"Yes Data Leakage"` / `"No Data Leakage"` strings and normalizing to boolean `is_leaking`.
- `DataUsageStatus`: Data ingestion auditor (`all_data_used: bool`, `missing_sources: list[str]`, `usage_recommendations: str`, `improved_code_block: Optional[str]`).

### E. Code Helper Utilities (`src/problem_2_v2/contracts/code_utils.py`)
- `extract_python_code(text: str) -> str`: Robustly strips markdown triple backticks (```python ... ```) and leading/trailing noise.
- `validate_python_syntax(code: str) -> tuple[bool, Optional[str]]`: Checks code with `ast.parse` and returns syntax error if invalid.
- `compute_code_diff(old_code: str, new_code: str) -> str`: Produces clean unified diff string for iteration logs.

## 3. Non-Functional Requirements
- **Pydantic V2:** Strict validation with `ConfigDict(extra='forbid', validate_assignment=True)`.
- **Type Annotations:** 100% type-hinted code compatible with `mypy --strict`.
- **Serialization:** Full JSON serialization and deserialization support with `.model_dump_json()`.
- **Test Coverage:** >90% unit test coverage for all schemas, validators, edge-case parsing, and code utilities.

## 4. Acceptance Criteria
- [ ] All Pydantic models instantiate, validate valid inputs, and reject invalid inputs with descriptive `ValidationError`.
- [ ] MetricDirection properly handles both MAXIMIZE and MINIMIZE comparison logic and score delta calculations.
- [ ] TaskSpecification parses both full markdown problem descriptions and programmatic dict inputs.
- [ ] DataLeakageStatus and DataUsageStatus cleanly parse both paper-exact prompt string outputs and structured JSON.
- [ ] Code utilities reliably extract clean Python code from LLM markdown responses, validate AST syntax, and produce git diffs.
- [ ] Comprehensive unit test suite in `tests/test_contracts.py` passes with 100% green status.
