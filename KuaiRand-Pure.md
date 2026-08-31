# Task Specification: KuaiRand-Pure Recommender Ranking

**Task Name:** Recommender Systems
**Task Type:** RECOMMENDER_RANKING
**Metric Name:** mean(GAUC, nDCG@5)
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
All models, candidates, and ensembling stages must strictly evaluate validation performance using the official `evaluate.py` evaluation harness, it is model-agnostic — `evaluate(user_ids: Any, labels: Any, scores: Any) -> dict[str, Any]`, so any model can be scored with it:
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

### CRITICAL DATA LEAKAGE, FEATURE USAGE & EVALUATION CONSTRAINTS

**DATASET PARTITIONING & TEMPORAL SPLIT RULES**

You MUST split the dataset strictly based on the date column. DO NOT use random splitting or train_test_split.

- **Source Files Available:** 
  - `log_standard_4_08_to_4_21_pure.csv`
  - `log_standard_4_22_to_5_08_pure.csv`

- **Split Rules (Based on Date Column `date` / `timestamp`):**
  - **Train Set:** Dates `20220408` to `20220421` (inclusive).
  - **Validation Set:** Dates `20220422` to `20220428` (inclusive).
  - **Hidden Test Set:** Dates `20220429` to `20220508` (used ONLY during final submission generation).


**FEATURE ROLE DEFINITIONS & ANTI-LEAKAGE BOUNDARIES**

[GROUP A: PRIMARY TARGET (y)]
- 'long_view'

[GROUP B: POST-INTERACTION SIGNALS (STRICTLY BANNED FROM INPUT MATRIX X) may only be used as auxiliary MTL loss heads]
The following columns from 'log_xxx.csv' represent user feedback that occurs DURING or AFTER the video impression.
They DO NOT EXIST at recommendation time:
- 'is_click'
- 'is_like'
- 'is_follow'
- 'is_comment'
- 'is_forward'
- 'is_hate'
- 'play_time_ms'
- 'profile_stay_time'
- 'comment_stay_time'
- 'is_profile_enter'

RULE:
- NEVER include any Group B column in the input feature tensor/matrix X.
- Group B columns MAY ONLY be used as auxiliary targets for Multi-Task Learning loss heads (e.g., auxiliary BCE on 'is_like' or regression loss on 'play_time_ms') during model training.
- Predictions for test submission must be based purely on model inference using allowed features (Group C, D, E).

[GROUP C: ALLOWED INTERACTION LOG CONTEXT (X_interaction)]
- 'user_id'
- 'video_id'
- 'tab'
- 'hourmin'
- 'is_rand'
- duration_ms 

[GROUP D (ALL ALLOWED): USER SIDE FEATURES (X_user, from user_features_pure.csv)]
Join strictly on 'user_id':
- Categorical / Discrete: 'user_active_degree', 'is_lowactive_period', 'is_live_streamer', 'is_video_author'
- Numerical / Ranges: 'follow_user_num', 'follow_user_num_range', 'fans_user_num', 'fans_user_num_range', 'friend_user_num', 'friend_user_num_range', 'register_days', 'register_days_range'
- One-Hot features: 'onehot_feat0' through 'onehot_feat17'

[GROUP E (ALL ALLOWED): VIDEO SIDE FEATURES (X_video)]
- Basic Metadata (video_features_basic_pure.csv)
- Prior Statistical Aggregations (video_features_statistic_pure.csv):
  Different from the basic features, the statistical features are the average statistics of the video each day over one month.

### Validated Empirical Insights
- **Negative Findings (Do not repeat):**
  - Adding static features (e.g. CWM 13 domains) yields no gain (0.5940 vs 0.5950), because `user_id x video_id` already captures most static signals.
  - First-order pure user features contribute zero to within-user ranking since they are constant within each user's candidate list.
  - Increasing FM latent dimension (k=8, 16, 32) does not improve performance.

**Constraints:**
- **Data Leakage & Subsampling:**
  - Do not use external datasets.
  - Subsample training data to at most 30,000 samples during iterative experimentation. Full training is reserved for final artifact production.
  - If the primary metric is 1.0 or comes close to 1.0, do verify that evaluation() is being used properly.

- **Convergence Rule:**
  - Convergence is defined as validation improvement delta <= epsilon = 0.002 over N = 3 consecutive iterations.


