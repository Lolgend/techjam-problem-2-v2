import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from evaluate import evaluate

np.random.seed(42)
torch.manual_seed(42)


def parse_dates(df):
    if 'date' in df.columns:
        s = df['date'].astype(str).str.replace('-', '').str.replace('/', '')
        df['date'] = s.str.slice(0, 8).astype(int)


class SASRec(nn.Module):
    def __init__(self, n_items, max_len=50, embed_dim=32, n_heads=4, n_layers=2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.drop = nn.Dropout(0.2)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                embed_dim, n_heads, dim_feedforward=128, dropout=0.2, batch_first=True
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.causal_mask = None

    def forward(self, seq, mask, cand_item):
        L = seq.size(1)
        x = self.item_emb(seq) + self.pos_emb(torch.arange(L, device=seq.device))[None]
        x = self.drop(x)

        if self.causal_mask is None or self.causal_mask.size(0) != L:
            self.causal_mask = torch.triu(
                torch.full((L, L), float('-inf'), device=seq.device), diagonal=1
            )

        for blk in self.blocks:
            x = blk(x, src_mask=self.causal_mask, src_key_padding_mask=~mask)

        h = self.norm(x)[:, -1, :]
        return torch.sigmoid((h * self.item_emb(cand_item)).sum(-1))


train_df = pd.read_csv('./input/log_standard_4_08_to_4_21_pure.csv')
valtest_df = pd.read_csv('./input/log_standard_4_22_to_5_08_pure.csv')

parse_dates(train_df)
parse_dates(valtest_df)

train_df = train_df[(train_df['date'] >= 20220408) & (train_df['date'] <= 20220421)]
val_df = valtest_df[(valtest_df['date'] >= 20220422) & (valtest_df['date'] <= 20220428)]

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

time_cols = ['date'] + [c for c in ['hour', 'minute', 'second'] if c in train_df.columns]

all_videos = set(train_df['video_id']).union(set(val_df['video_id']))
vid2idx = {vid: i + 1 for i, vid in enumerate(all_videos)}
n_items = len(all_videos)

sampled = train_df.sample(n=min(30000, len(train_df)), random_state=42)
sampled_idx = set(sampled.index)

max_len = 50
seqs = []
cands = []
labels = []
user_hist = {}

train_sorted = train_df.sort_values(['user_id'] + time_cols, kind='mergesort')

for user_id, group in train_sorted.groupby('user_id', sort=False):
    hist = []
    for row in group.itertuples():
        if row.Index in sampled_idx:
            seq = np.zeros(max_len, dtype=np.int64)
            if len(hist) > 0:
                take = hist[-max_len:]
                seq[-len(take):] = take
            seqs.append(seq)
            cands.append(vid2idx[row.video_id])
            labels.append(int(row.long_view))

        hist.append(vid2idx[row.video_id])
        if len(hist) > max_len:
            hist = hist[-max_len:]

    user_hist[user_id] = hist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SASRec(n_items, max_len=max_len, embed_dim=32, n_heads=4, n_layers=2)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()
batch_size = 512
epochs = 2

model.train()
for epoch in range(epochs):
    perm = np.random.permutation(len(seqs))
    for i in range(0, len(seqs), batch_size):
        idx = perm[i:i + batch_size]

        seq_batch = torch.from_numpy(np.stack([seqs[j] for j in idx])).to(device)
        cand_batch = torch.from_numpy(np.array([cands[j] for j in idx])).to(device)
        label_batch = torch.from_numpy(np.array([labels[j] for j in idx], dtype=np.float32)).to(device)

        mask = seq_batch != 0
        mask[:, -1] = True

        pred = model(seq_batch, mask, cand_batch)
        loss = loss_fn(pred, label_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()

val_user_ids = []
val_labels = []
val_preds = []
val_order = []

val_sorted = val_df.sort_values(['user_id'] + time_cols, kind='mergesort')

with torch.no_grad():
    for user_id, group in val_sorted.groupby('user_id', sort=False):
        hist = list(user_hist.get(user_id, []))

        for row in group.itertuples():
            seq = np.zeros(max_len, dtype=np.int64)
            if len(hist) > 0:
                take = hist[-max_len:]
                seq[-len(take):] = take

            seq_t = torch.from_numpy(seq).to(device).unsqueeze(0)
            mask_t = seq_t != 0
            mask_t[:, -1] = True

            cand_t = torch.from_numpy(np.array([vid2idx[row.video_id]])).to(device)

            prob = model(seq_t, mask_t, cand_t).item()

            val_user_ids.append(user_id)
            val_labels.append(int(row.long_view))
            val_preds.append(prob)
            val_order.append(row.Index)

            hist.append(vid2idx[row.video_id])
            if len(hist) > max_len:
                hist = hist[-max_len:]

order = np.argsort(val_order)
val_user_ids = [val_user_ids[i] for i in order]
val_labels = [val_labels[i] for i in order]
val_preds = [val_preds[i] for i in order]

val_res = evaluate(val_user_ids, val_labels, val_preds)
print(f'Final Validation Performance: {val_res["primary"]}')