# Task Specification: KuaiRand-Pure Recommender Ranking

**Task Name:** KuaiRand-Pure Recommender Ranking
**Task Type:** RECOMMENDER_RANKING
**Metric Name:** primary (mean(GAUC, nDCG@5))
**Metric Direction:** MAXIMIZE
**Target Variable:** long_view
**Baseline Score:** 0.6016
**Subsample Size:** 30000
**Dataset Files:** log_standard_4_08_to_4_21_pure.csv, log_standard_4_22_to_5_08_pure.csv, video_features_basic_pure.csv, user_features_pure.csv, video_features_statistic_pure.csv

**Description:**
The goal of this task is within-user ranking over video impression logs from the KuaiRand-Pure dataset.
Each user only has their impressions ranked in the evaluation split (no full-catalog retrieval is needed).
The relevance target label is `long_view` (binary 0 or 1).

### Evaluation Metrics
The primary evaluation metric is the arithmetic mean of Group AUC (GAUC) and normalized Discounted Cumulative Gain at rank 5 (nDCG@5):
`primary = (GAUC + nDCG@5) / 2.0`

This metric is computed by the official evaluation harness `src/baseline/evaluate.py`
(`evaluate.evaluate(user_ids, labels, scores, k=5)`), which returns a dict with
keys `GAUC`, `nDCG@5`, `primary`, `users`, and `rows`. Do **not** reimplement the
metric — import `evaluate` in your sandboxed script (the module is importable because
the workspace root and `src/baseline` are injected into the subprocess `PYTHONPATH`).
The exact calculation rules (write-in-stone, per `evaluate.py`):

1. **GAUC (Group AUC):** Computed with the Mann-Whitney U statistic (tie-corrected,
   equivalent to `sklearn.metrics.roc_auc_score`). Only users with
   `0 < positive_count < impression_count` contribute, weighted by the number of
   positive impressions per user. If no eligible user exists, GAUC defaults to `0.5`.
2. **nDCG@5:** Computed per user with gain `2^rel - 1` (equivalent to identity under
   binary labels). All-negative users (27.1% of the dataset) receive an nDCG score of
   `0.0` and are included in the mean. All-positive users (9.2%) receive `1.0`.

### Baseline Ladder & Oracle Headroom
- **Random Baseline (Lower Bound):** Validation primary = 0.4834 | Test primary = 0.4753
- **Item Popularity Baseline:** Validation primary = 0.5807 | Test primary = 0.5715
- **Official FM Baseline (Target to Beat):** Validation primary = 0.6016 | Test primary = 0.5946 (std over 5 seeds = 0.0008)
- **Oracle Theoretical Ceiling:** Validation primary = 0.8484 | Test primary = 0.8645 (nDCG ceiling is 0.7289 due to 27.1% all-negative users)

### Validated Empirical Insights & Search Directions
- **Negative Findings (Do not repeat):**
  - Adding static features (e.g. CWM 13 domains) yields no gain (0.5940 vs 0.5950), because `user_id x video_id` already captures most static signals.
  - First-order pure user features contribute zero to within-user ranking since they are constant within each user's candidate list.
  - Increasing FM latent dimension (k=8, 16, 32) does not improve performance.
- **High-Headroom Directions to Explore:**
  - **Ranking Loss Functions:** Pairwise (BPR / Margin ranking) or listwise (per-user softmax) objectives aligned with GAUC/nDCG.
  - **Sequential User History Modeling:** Utilizing user interaction sequences via attention architectures (DIN, SASRec, SIM).
  - **Auxiliary Multi-Task Learning:** Leveraging auxiliary interactions (`is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `play_time_ms`).
  - **Censored Watch-Time Modeling:** Modeling duration with survival/censoring regression.
  - **Distribution Shift & Temporal Dynamics:** Temporal features (`date`, `hour`, recency) between train and test periods.

**Constraints:**
- **Data Splits (Deterministic chronological split):**
  - Train period: `20220408` to `20220421` (from `log_standard_4_08_to_4_21_pure.csv`).
  - Validation period: `20220422` to `20220428` (from `log_standard_4_22_to_5_08_pure.csv`).
  - Test period: `20220429` to `20220508` (from `log_standard_4_22_to_5_08_pure.csv`).
- **Data Leakage & Subsampling:**
  - Do not use external datasets.
  - Subsample training data to at most 30,000 samples during iterative experimentation. Full training is reserved for final artifact production.
  - Guard against target leakage in categorical and target encodings (use out-of-fold / leave-one-out with shrinkage).
- **Submission File Format:**
  - Must be a CSV file with exact header: `row_id,user_id,video_id,score`
    (the exact 4-column schema `["row_id", "user_id", "video_id", "score"]`
    required by `src/baseline/submit.py`, `submit.HEADER`).
  - `row_id`: 0-indexed contiguous integer matching the exact deterministic row
    order of `data.load()["test"]` from `src/baseline/data.py` — which reads
    `log_standard_4_08_to_4_21_pure.csv` first, then
    `log_standard_4_22_to_5_08_pure.csv`, filters rows by the test date range, and
    preserves the original file order.
  - `user_id` and `video_id`: Redundant validation fields strictly aligned with test
    set rows. `(user_id, video_id)` pairs are not unique (3.06% duplicate pairs in
    test), making `row_id` necessary as the true primary key.
  - `score`: Continuous real-valued model prediction (no NaN or Inf).
  - Submissions are verified with the official harness
    `python src/baseline/submit.py --check submission.csv --data_dir <data>`; it
    rejects misaligned rows, non-contiguous `row_id`, wrong field counts, and
    non-finite scores.
- **Convergence Rule:**
  - Convergence is defined as validation improvement delta <= epsilon = 0.002 over N = 3 consecutive iterations.
- **Output Validation Score Standard:**
  - All executable solution scripts must output the validation score to stdout in the exact format:
    `Final Validation Performance: {final_validation_score}`
