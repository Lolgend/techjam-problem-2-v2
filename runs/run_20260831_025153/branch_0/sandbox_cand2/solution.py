import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import deque
from evaluate import evaluate


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DIN(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=16, feat_dim=20, max_hist=50):
        super(DIN, self).__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        self.hist_emb = nn.Embedding(n_items, embed_dim)

        self.attn = nn.Sequential(
            nn.Linear(embed_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3 + feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.bn = nn.BatchNorm1d(feat_dim)

    def forward(self, user, item, hist, hist_mask, feat):
        u = self.user_emb(user)
        i = self.item_emb(item)
        h = self.hist_emb(hist)

        candidate = i.unsqueeze(1).expand_as(h)
        attn_in = torch.cat([h, candidate, h * candidate], dim=-1)
        attn_logits = self.attn(attn_in).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~hist_mask, -1e9)

        weights = torch.softmax(attn_logits, dim=-1)
        weights = weights * hist_mask.float()
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)

        interest = (weights.unsqueeze(-1) * h).sum(dim=1)

        x = torch.cat([u, i, interest, self.bn(feat)], dim=-1)
        return self.mlp(x).squeeze(-1)


set_seed(42)

# ---------- Load logs ----------
train_log = pd.read_csv('./input/log_standard_4_08_to_4_21_pure.csv')
future_log = pd.read_csv('./input/log_standard_4_22_to_5_08_pure.csv')

train_log['date_str'] = train_log['date'].astype(str).str.extract(r'(\d{8})')[0]
future_log['date_str'] = future_log['date'].astype(str).str.extract(r'(\d{8})')[0]

train_log = train_log[
    (train_log['date_str'] >= '20220408') & (train_log['date_str'] <= '20220421')
].copy()

val_log = future_log[
    (future_log['date_str'] >= '20220422') & (future_log['date_str'] <= '20220428')
].copy()

if len(train_log) > 30000:
    train_log = train_log.sample(n=30000, random_state=42).reset_index(drop=True)

# ---------- Load features ----------
user_feat = pd.read_csv('./input/user_features_pure.csv')
video_basic = pd.read_csv('./input/video_features_basic_pure.csv')
video_stat = pd.read_csv('./input/video_features_statistic_pure.csv')

train = train_log.merge(user_feat, on='user_id', how='left')
train = train.merge(video_basic, on='video_id', how='left')
train = train.merge(video_stat, on='video_id', how='left')

val = val_log.merge(user_feat, on='user_id', how='left')
val = val.merge(video_basic, on='video_id', how='left')
val = val.merge(video_stat, on='video_id', how='left')

# ---------- Build feature columns ----------
blacklist = {'user_id', 'video_id', 'long_view', 'is_click', 'is_like',
             'is_follow', 'is_comment', 'is_forward', 'play_time_ms',
             'date', 'date_str'}


def numeric_cols(df, id_col):
    cols = []
    for c in df.columns:
        if c == id_col or c in blacklist:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


feat_cols = numeric_cols(user_feat, 'user_id')
feat_cols += numeric_cols(video_basic, 'video_id')
feat_cols += numeric_cols(video_stat, 'video_id')
feat_cols = list(dict.fromkeys(feat_cols))

if 'hour' in train.columns and pd.api.types.is_numeric_dtype(train['hour']) and 'hour' not in feat_cols:
    feat_cols.append('hour')

feat_cols = [c for c in feat_cols if c in train.columns and c in val.columns]

if len(feat_cols) == 0:
    feat_cols = ['dummy']
    train['dummy'] = 0.0
    val['dummy'] = 0.0

# ---------- ID maps ----------
train_users = train_log['user_id'].unique()
train_items = train_log['video_id'].unique()

user2idx = {u: i + 1 for i, u in enumerate(train_users)}
item2idx = {v: i + 1 for i, v in enumerate(train_items)}

n_users = len(train_users) + 1
n_items = len(train_items) + 1

train['u_idx'] = train['user_id'].map(user2idx).fillna(0).astype(np.int64)
train['i_idx'] = train['video_id'].map(item2idx).fillna(0).astype(np.int64)

val['u_idx'] = val['user_id'].map(user2idx).fillna(0).astype(np.int64)
val['i_idx'] = val['video_id'].map(item2idx).fillna(0).astype(np.int64)

# ---------- Build sequential histories ----------
max_hist = 30

sort_cols = ['user_id', 'date_str']
for c in ['hour', 'minute', 'second']:
    if c in train.columns:
        sort_cols.append(c)

train = train.sort_values(sort_cols).reset_index(drop=True)

train_hist = np.zeros((len(train), max_hist), dtype=np.int64)
train_hist_mask = np.zeros((len(train), max_hist), dtype=np.bool_)

hist_buf = {}

for pos, row in enumerate(train.itertuples(index=False)):
    u = int(row.u_idx)
    if u not in hist_buf:
        hist_buf[u] = deque(maxlen=max_hist)

    hist = hist_buf[u]
    hist_list = list(hist)
    k = len(hist_list)

    if k > 0:
        train_hist[pos, max_hist - k:] = hist_list
        train_hist_mask[pos, max_hist - k:] = True

    hist.append(int(row.i_idx))

val_hist = np.zeros((len(val), max_hist), dtype=np.int64)
val_hist_mask = np.zeros((len(val), max_hist), dtype=np.bool_)

for pos, row in enumerate(val.itertuples(index=False)):
    u = int(row.u_idx)
    hist_list = list(hist_buf.get(u, deque()))[-max_hist:]
    k = len(hist_list)
    if k > 0:
        val_hist[pos, max_hist - k:] = hist_list
        val_hist_mask[pos, max_hist - k:] = True

# ---------- Feature matrices ----------
train_feat = train[feat_cols].fillna(0).to_numpy(dtype=np.float32)
val_feat = val[feat_cols].fillna(0).to_numpy(dtype=np.float32)

feat_mean = train_feat.mean(axis=0)
feat_std = train_feat.std(axis=0) + 1e-8

train_feat = (train_feat - feat_mean) / feat_std
val_feat = (val_feat - feat_mean) / feat_std

# ---------- Tensors ----------
train_u = train['u_idx'].to_numpy(dtype=np.int64)
train_i = train['i_idx'].to_numpy(dtype=np.int64)
train_y = train['long_view'].to_numpy(dtype=np.float32)

val_u = val['u_idx'].to_numpy(dtype=np.int64)
val_i = val['i_idx'].to_numpy(dtype=np.int64)
val_y = val['long_view'].to_numpy(dtype=np.float32)

val_user_ids = val['user_id'].tolist()
val_labels = val_y.astype(int).tolist()

# ---------- Model ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DIN(
    n_users=n_users,
    n_items=n_items,
    embed_dim=16,
    feat_dim=len(feat_cols),
    max_hist=max_hist
).to(device)

with torch.no_grad():
    model.user_emb.weight.data[0] = 0.0
    model.item_emb.weight.data[0] = 0.0
    model.hist_emb.weight.data[0] = 0.0

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()

epochs = 5
batch_size = 256

model.train()
for epoch in range(epochs):
    perm = np.random.permutation(len(train_u))
    for i in range(0, len(train_u), batch_size):
        idx = perm[i:i + batch_size]

        user = torch.tensor(train_u[idx], dtype=torch.long, device=device)
        item = torch.tensor(train_i[idx], dtype=torch.long, device=device)
        hist = torch.tensor(train_hist[idx], dtype=torch.long, device=device)
        hist_mask = torch.tensor(train_hist_mask[idx], dtype=torch.bool, device=device)
        feat = torch.tensor(train_feat[idx], dtype=torch.float32, device=device)
        y = torch.tensor(train_y[idx], dtype=torch.float32, device=device)

        logits = model(user, item, hist, hist_mask, feat)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ---------- Validation predictions ----------
model.eval()
val_preds = []

with torch.no_grad():
    for i in range(0, len(val_u), batch_size):
        user = torch.tensor(val_u[i:i + batch_size], dtype=torch.long, device=device)
        item = torch.tensor(val_i[i:i + batch_size], dtype=torch.long, device=device)
        hist = torch.tensor(val_hist[i:i + batch_size], dtype=torch.long, device=device)
        hist_mask = torch.tensor(val_hist_mask[i:i + batch_size], dtype=torch.bool, device=device)
        feat = torch.tensor(val_feat[i:i + batch_size], dtype=torch.float32, device=device)

        logits = model(user, item, hist, hist_mask, feat)
        preds = torch.sigmoid(logits).cpu().numpy()
        val_preds.append(preds)

val_preds = np.concatenate(val_preds)
val_predictions = val_preds.tolist()

# ---------- Evaluate ----------
val_res = evaluate(val_user_ids, val_labels, val_predictions)
print(f'Final Validation Performance: {val_res["primary"]:.6f}')