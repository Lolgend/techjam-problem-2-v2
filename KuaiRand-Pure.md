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
The goal of this task is within-user ranking over logged video impressions from the KuaiRand-Pure dataset.
Each user only has their impressions ranked in the evaluation split (no full-catalog retrieval is needed).
The relevance target label is `long_view` (binary 0 or 1).

### Evaluation Protocol & Metric Tooling (MANDATORY)
All models, candidates, and ensembling stages must strictly evaluate validation performance using the official `evaluate.py` evaluation harness, it is model-agnostic — it takes only (user_ids, labels, scores), so any model can be scored with it:
```python
from evaluate import evaluate

# user_ids: sequence of validation user IDs
# labels: sequence of validation binary labels (0 or 1)
# scores: continuous real-valued prediction scores from model
val_res = evaluate(val_user_ids, val_labels, val_predictions)
print(f"Final Validation Performance: {val_res['primary']:.6f}")
```

The metric `primary` is the arithmetic mean of Group AUC (GAUC) and normalized Discounted Cumulative Gain at rank 5 (nDCG@5):
`primary = (GAUC + nDCG@5) / 2.0`
- **GAUC:** Evaluated strictly on discriminative users where `0 < positive_count < impression_count`, weighted by positive impressions.
- **nDCG@5:** Evaluated per user with gain `2^rel - 1`. All-negative users (27.1% of dataset) receive 0.0 and are included in the mean.

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
  - Develop on train + validation only; the hidden test set is scored once. 
  - Splitting by date rather than by row count avoids any tie-breaking ambiguity on equal timestamps.
- **Data Leakage & Subsampling:**
  - Do not use external datasets.
  - Subsample training data to at most 30,000 samples during iterative experimentation. Full training is reserved for final artifact production.
  - Guard against target leakage in categorical and target encodings (use out-of-fold / leave-one-out with shrinkage).
- **Mandatory Evaluation & Submission Standards:**
  - Every solution script must evaluate validation predictions with `evaluate()` from `evaluate.py`.
  - The final production script must write `./final/submission.csv` with exact columns: `row_id,user_id,video_id,score` and must pass verification with `python3 submit.py --check ./final/submission.csv`.
- **Convergence Rule:**
  - Convergence is defined as validation improvement delta <= epsilon = 0.002 over N = 3 consecutive iterations.
- **Output Validation Score Standard:**
  - All executable solution scripts must output the validation score to stdout in the exact format:
    `Final Validation Performance: {final_validation_score}`
