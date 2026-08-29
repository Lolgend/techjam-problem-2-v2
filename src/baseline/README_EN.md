# KuaiRand-Pure Starter Kit

## Dependencies

Python 3.9+ and numpy. **Nothing else.** torch, pandas, and sklearn are not required.

## Data

Download the data from https://kuairand.com (direct Zenodo link; no registration required):

```bash
# Run in the Starter Kit directory. This extracts ./KuaiRand-Pure/
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## Running

```bash
python3 baseline.py --model fm
```

`--data_dir` defaults to `./KuaiRand-Pure/data`. Specify it explicitly if the data is stored elsewhere.

`--model` can be `fm` (the official baseline), `pop` (the trivial baseline), or `random` (the lower bound, useful for checking the evaluation code).
The FM baseline takes about 40 seconds on a single CPU core.

## Task Definition (Fixed; Do Not Change)

| | |
|---|---|
| Task | **Within-user ranking** - rank only the exposures shown to each user in the evaluation split; do not retrieve from the full catalog |
| Relevance label | `long_view` (native 0/1 column) |
| Metrics | `GAUC` and `nDCG@5`; **primary score = their average** |
| Data splits | train `20220408-20220421` / valid `20220422-20220428` / test `20220429-20220508` |
| Users with zero positive examples | nDCG is 0.0 and is included in the average; GAUC includes only users with `0 < positive_count < exposure_count`, weighted by positive count |
| nDCG gain | `2^rel - 1` (equivalent to identity for binary labels) |

The implementation is in `evaluate.py`; all conventions are documented in its header comments.

## Baseline Ladder

Scores on the test set. **The target to beat is the FM row.**

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (lower bound; for self-checking) | 0.4996 | 0.4511 | 0.4753 |
| item popularity (trivial) | 0.6308 | 0.5121 | 0.5715 |
| **FM (official baseline)** | **0.6610** | **0.5282** | **0.5946** |

### Important: The True nDCG@5 Range Tops Out at 0.729, Not 1.0

Among the 23,875 users in the test set:

| | Share | Effect on the metric |
|---|---|---|
| All-negative users (none of the user's exposures are long views) | **27.1%** | nDCG is always **0**; no model can improve it, and these users are excluded from GAUC |
| All-positive users | **9.2%** | nDCG is always **1**; these users are excluded from GAUC |
| Users with meaningful distinctions | **63.7%** | The effective sample for GAUC |

Therefore, even using the true labels as prediction scores (an oracle with perfect ranking) only achieves:

| | random | FM baseline | **oracle upper bound** | FM share of available range |
|---|---|---|---|---|
| GAUC | 0.4996 | 0.6610 | **1.0000** | 32.3% |
| nDCG@5 | 0.4511 | 0.5282 | **0.7289** | 27.8% |
| **primary** | 0.4753 | **0.5946** | **0.8645** | **30.7%** |

**Use the oracle as the denominator when tracking progress.** Treating 0.5946 as being far from a perfect 1.0 is misleading: the baseline has already captured 30% of the usable range, so the remaining headroom is 0.27 rather than 0.41.

Across five random seeds, the FM baseline has a standard deviation of 0.0008. The convergence criterion is therefore `epsilon = 0.002` (about 2.5 standard deviations), `N = 3`:
consider the model converged when the validation primary score improves by no more than 0.002 for three consecutive iterations.

> Self-check: if `--model random` does not produce a primary score near 0.475 (within +/-0.001), the evaluation harness has a problem. Fix it first.

## Submission Format

CSV with a header, containing one row for each row in the evaluation split:

```csv
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
...
```

| Field | Description |
|---|---|
| `row_id` | Consecutive integers starting at 0, matching the row order of `data.load()[split]`. The order is deterministic: read `log_standard_4_08_to_4_21_pure.csv` first, then `log_standard_4_22_to_5_08_pure.csv`, filter by date, and preserve file order. |
| `user_id` / `video_id` | Redundant fields used only to verify alignment |
| `score` | The score assigned by your model to that row. Any real number is valid; only relative ordering matters. NaN and Inf are not allowed. |

> **Why `row_id` is required:** `(user_id, video_id)` is **not unique** in the evaluation set. The test set contains 3.06% duplicate pairs, with up to 12 occurrences of the same pair. Therefore, it cannot be used as the primary key.

Generate and validate a submission:

```bash
python3 submit.py --make  --split test  submission.csv    # Generate an example submission with the official FM baseline
python3 submit.py --check --split test  submission.csv    # Validate format and alignment
python3 submit.py --score --split valid submission.csv    # Validate and score (local validation data is available)
```

`--check` rejects an incorrect header, the wrong number of rows, skipped `row_id` values, misaligned `user_id` or `video_id` values, non-numeric scores, and NaN or Inf scores. **Always run `--check` before submitting.**

## Where to Start

The ordering below is **based on measurements**, not speculation. The organizers have marked known dead ends so you do not spend iterations repeating them.

### Already Tested: These Two Approaches Did Not Help

| Tested approach | Result |
|---|---|
| **Adding static features** - adding all 13 CWM feature fields (`music_id`, `video_type`, `upload_type`, and six coarse user-side buckets) | primary **0.5940** vs. **0.5950** with five fields; no meaningful difference within noise, and slightly lower |
| **Increasing model capacity** - embedding dimensions `k = 8 / 16 / 32` | 0.5895 / 0.5902 / 0.5887; effectively unchanged |

The `user_id x video_id` interaction already captures most of the learnable signal. Coarse buckets such as `follow_user_num_range` are redundant in the presence of `user_id`, and 1.14 million rows are not enough to support substantially larger capacity. **The bottleneck is not features or capacity.**

Also note: **a first-order term containing only user-side features contributes a constant within each user and therefore has no effect on ranking.** In-user ranking is invariant to any within-user constant (in experiments, `item_pop x user_bias` produces exactly the same scores as plain `item_pop`). User-side features can only help through **interaction terms with item-side features**.

### Unexplored: Where the Headroom Is Likely to Be

The following possibilities are listed in our estimated order of potential (**the organizers have not tested these; they are left for you**):

1. **Change the loss function.** The current objective is pointwise log loss, but the metrics (GAUC / nDCG) are ranking metrics. Try a pairwise objective such as BPR or a listwise objective such as a softmax over each user's exposures. Aligning the training objective with the evaluation protocol is the most promising direction in our view.
2. **User history sequences.** The current features use **no behavior sequence at all**. Each KuaiRand user has hundreds to thousands of training interactions, leaving DIN/SIM-style interest modeling unexplored.
3. **Multi-objective learning.** The logs also contain `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, and `play_time_ms`, which can provide auxiliary tasks for the main `long_view` task.
4. **Watch-time modeling.** This is the focus of CWM: it models watch time with **censored regression**. When a video finishes, the observed watch time is truncated, so a one-sided loss is used instead of squared error. This is a research-oriented direction with substantial depth.
5. **Change the model.** DeepFM / DCN / xDeepFM. Since capacity was not the bottleneck in the experiments, prioritize this after items 1-4.
6. **Time features and distribution shift.** Consider `hourmin`, `date`, and the shift between the train and test distributions.
7. **Unbiased validation (advanced).** `log_random_4_22_to_5_08_pure.csv` is a random-exposure log (1.18 million rows) that can serve as an additional unbiased validation set, helping check whether the model is overfitting biased traffic.

## Using Your Own Model (Including CWM)

`evaluate.py` is completely decoupled from the model. It only requires three arrays of equal length:

```python
from evaluate import evaluate

print(evaluate(user_ids, labels, scores))  # scores can come from any model
```

- `user_ids`: the user ID for each row in the evaluation split
- `labels`: the row's `long_view` value (0/1)
- `scores`: the score assigned by your model to each row (any real number; only relative ordering matters)

You can therefore ignore `baseline.py` entirely and use PyTorch, LightGBM, or the xDeepFM implementation from [CWM](https://github.com/hyz20/CWM). As long as you pass the resulting `scores` to `evaluate()`, **the scoring protocol is defined exclusively by `evaluate.py`.**

> CWM note: it depends on `torch==1.6.0` (released in 2020), which may be difficult to install on newer GPUs. Its loss optimizes counterfactual watch time, while the evaluation label is a reconstructed `long_view2`. It is research code for watch-time debiasing and is best treated as an advanced reference rather than a starting point.

## Files

| File | Description |
|---|---|
| `evaluate.py` | Metric implementation and all evaluation conventions. **Do not modify.** |
| `data.py` | Data loading, official splits, and feature encoding. Modify this to add features. |
| `baseline.py` | Three baselines. FM is the one to beat. |
| `baseline_scores.json` | Official scores, seed variance, and convergence parameters. |
| `submit.py` | Submission generation and validation. |
| `ablation_features.py` | Feature ablation experiments reproducing the result that adding features did not help. |
