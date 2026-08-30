import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from evaluate import evaluate

np.random.seed(42)
torch.manual_seed(42)


def parse_date(df):
    if 'date' not in df:
        return df
    if pd.api.types.is_numeric_dtype(df['date']):
        df['date'] = df['date'].astype(int)
    else:
        sample = str(df['date'].iloc[0])
        if '-' in sample or '/' in sample or ':' in sample:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d').astype(int)
        else:
            df['date'] = pd.to_numeric(df['date'], errors='coerce').astype(int)
    return df


def id_to_str(series):
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(-1).astype(np.int64).astype(str)
    return series.astype(str)


train_full = pd.read_csv('./input/log_standard_4_08_to_4_21_pure.csv')
future_full = pd.read_csv('./input/log_standard_4_22_to_5_08_pure.csv')

train_full = parse_date(train_full)
future_full = parse_date(future_full)

video_pop = train_full.groupby('video_id')['long_view'].agg(['mean', 'count']).reset_index()
video_pop.columns = ['video_id', 'item_pop', 'item_count']

if len(train_full) > 30000:
    train_df = train_full.sample(n=30000, random_state=42).reset_index(drop=True)
else:
    train_df = train_full.reset_index(drop=True)

val_df = future_full[future_full['date'] <= 20220428].reset_index(drop=True)

train_df['user_id'] = id_to_str(train_df['user_id'])
train_df['video_id'] = id_to_str(train_df['video_id'])
val_df['user_id'] = id_to_str(val_df['user_id'])
val_df['video_id'] = id_to_str(val_df['video_id'])
video_pop['video_id'] = id_to_str(video_pop['video_id'])

train_df = train_df.merge(video_pop, on='video_id', how='left')
val_df = val_df.merge(video_pop, on='video_id', how='left')

stat_df = pd.read_csv('./input/video_features_statistic_pure.csv')
stat_df['video_id'] = id_to_str(stat_df['video_id'])
exclude_cols = {'video_id', 'author_id', 'music_id', 'upload_type', 'video_type', 'music_type'}
stat_cols = []
for c in stat_df.columns:
    if c in exclude_cols:
        continue
    if c.lower().endswith('_id'):
        continue
    if stat_df[c].dtype in ['int64', 'float64']:
        stat_cols.append(c)
stat_cols = stat_cols[:20]

train_df = train_df.merge(stat_df[['video_id'] + stat_cols], on='video_id', how='left')
val_df = val_df.merge(stat_df[['video_id'] + stat_cols], on='video_id', how='left')


def build_features(df):
    if 'hour' in df.columns:
        hour = pd.to_numeric(df['hour'], errors='coerce').fillna(0)
    else:
        hour = pd.Series(0, index=df.index)
    feat = pd.DataFrame(index=df.index)
    feat['hour_sin'] = np.sin(2.0 * np.pi * hour / 24.0)
    feat['hour_cos'] = np.cos(2.0 * np.pi * hour / 24.0)
    feat['day_since_start'] = (pd.to_numeric(df['date'], errors='coerce') - 20220408).fillna(0)
    if 'item_pop' in df.columns:
        feat['item_pop'] = pd.to_numeric(df['item_pop'], errors='coerce').fillna(0)
    if 'item_count' in df.columns:
        feat['item_count'] = np.log1p(pd.to_numeric(df['item_count'], errors='coerce').fillna(0))
    for c in stat_cols:
        if c in df.columns:
            feat[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    return feat


train_feats = build_features(train_df)
val_feats = build_features(val_df)

scaler = StandardScaler()
train_feats_s = scaler.fit_transform(train_feats)
val_feats_s = scaler.transform(val_feats)

enc_df = pd.concat([train_df[['user_id', 'video_id']], val_df[['user_id', 'video_id']]], ignore_index=True)
user_enc = LabelEncoder().fit(enc_df['user_id'])
item_enc = LabelEncoder().fit(enc_df['video_id'])

train_users = user_enc.transform(train_df['user_id'])
train_items = item_enc.transform(train_df['video_id'])
val_users = user_enc.transform(val_df['user_id'])
val_items = item_enc.transform(val_df['video_id'])

train_labels = train_df['long_view'].astype(np.int32).values
val_labels = val_df['long_view'].astype(np.int32).values

n_users = len(user_enc.classes_)
n_items = len(item_enc.classes_)


class BPRRanker(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=16, feat_dim=20):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        self.feat_mlp = nn.Sequential(nn.Linear(feat_dim, 16), nn.ReLU())
        self.bias = nn.Parameter(torch.zeros(1))

    def score(self, user, item, feat):
        return (self.user_emb(user) * self.item_emb(item)).sum(-1) + self.feat_mlp(feat).sum(-1) + self.bias


model = BPRRanker(n_users, n_items, embed_dim=16, feat_dim=train_feats_s.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

pos_dict = defaultdict(list)
neg_dict = defaultdict(list)
for i in range(len(train_df)):
    if train_labels[i] == 1:
        pos_dict[train_users[i]].append(i)
    else:
        neg_dict[train_users[i]].append(i)

users_with_both = [u for u in pos_dict if u in neg_dict]

epochs = 5
iterations_per_epoch = 200
batch_size = 512

for epoch in range(epochs):
    model.train()
    for _ in range(iterations_per_epoch):
        users = np.random.choice(users_with_both, size=batch_size, replace=True)
        pos_idx = [int(np.random.choice(pos_dict[u])) for u in users]
        neg_idx = [int(np.random.choice(neg_dict[u])) for u in users]

        user_t = torch.tensor(users, dtype=torch.long)
        pos_item_t = torch.tensor(train_items[pos_idx], dtype=torch.long)
        neg_item_t = torch.tensor(train_items[neg_idx], dtype=torch.long)
        pos_feat_t = torch.tensor(train_feats_s[pos_idx], dtype=torch.float32)
        neg_feat_t = torch.tensor(train_feats_s[neg_idx], dtype=torch.float32)

        optimizer.zero_grad()
        pos_score = model.score(user_t, pos_item_t, pos_feat_t)
        neg_score = model.score(user_t, neg_item_t, neg_feat_t)
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
        loss.backward()
        optimizer.step()


def predict(model, users, items, feats, batch_size=1024):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(users), batch_size):
            u = torch.tensor(users[i:i + batch_size], dtype=torch.long)
            it = torch.tensor(items[i:i + batch_size], dtype=torch.long)
            f = torch.tensor(feats[i:i + batch_size], dtype=torch.float32)
            preds.append(model.score(u, it, f).numpy())
    return np.concatenate(preds)


val_preds = predict(model, val_users, val_items, val_feats_s)

val_user_ids = val_df['user_id'].tolist()
val_res = evaluate(val_user_ids, val_labels.tolist(), val_preds.tolist())
final_score = val_res['primary']
print(f'Final Validation Performance: {final_score}')