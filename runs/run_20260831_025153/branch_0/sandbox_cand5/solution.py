import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from evaluate import evaluate


def parse_date_to_int(x):
    if pd.isna(x):
        return None
    s = str(x).strip().split(' ')[0]
    s = s.replace('-', '').replace('/', '')
    if s.isdigit() and len(s) >= 8:
        return int(s[:8])
    return None


class MultiTaskSharedBottom(nn.Module):
    def __init__(self, n_users, n_items, feat_dim, embed_dim=16, n_aux=5):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        self.shared = nn.Sequential(
            nn.Linear(embed_dim * 2 + feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.main = nn.Linear(128, 1)
        self.aux_heads = nn.ModuleList([nn.Linear(128, 1) for _ in range(n_aux)])
        self.watch_head = nn.Linear(128, 1)

    def forward(self, user, item, feat):
        h = torch.cat([self.user_emb(user), self.item_emb(item), feat], dim=-1)
        h = self.shared(h)
        main = torch.sigmoid(self.main(h)).squeeze(-1)
        aux = [torch.sigmoid(head(h)).squeeze(-1) for head in self.aux_heads]
        watch = self.watch_head(h).squeeze(-1)
        return main, aux, watch


def main():
    train_log = pd.read_csv('./input/log_standard_4_08_to_4_21_pure.csv')
    later_log = pd.read_csv('./input/log_standard_4_22_to_5_08_pure.csv')

    date_col = 'date' if 'date' in train_log.columns else ('day' if 'day' in train_log.columns else None)
    if date_col is None:
        raise ValueError('No date column found.')

    train_log['date_int'] = train_log[date_col].apply(parse_date_to_int)
    later_log['date_int'] = later_log[date_col].apply(parse_date_to_int)

    train_log = train_log[
        (train_log['date_int'] >= 20220408) & (train_log['date_int'] <= 20220421)
    ].reset_index(drop=True)

    val_log = later_log[
        (later_log['date_int'] >= 20220422) & (later_log['date_int'] <= 20220428)
    ].reset_index(drop=True)

    if len(train_log) > 30000:
        train_log = train_log.sample(n=30000, random_state=42).reset_index(drop=True)

    for df in [train_log, val_log]:
        df['user_id'] = df['user_id'].astype(str)
        df['video_id'] = df['video_id'].astype(str)
        if 'hour' not in df.columns:
            df['hour'] = 0
        else:
            df['hour'] = df['hour'].fillna(0)
        dt = pd.to_datetime(df['date_int'].astype(str), format='%Y%m%d', errors='coerce')
        df['weekday'] = dt.dt.weekday.fillna(0).astype(int)
        df['day_index'] = df['date_int'] - 20220408

    user_features = pd.read_csv('./input/user_features_pure.csv')
    video_basic = pd.read_csv('./input/video_features_basic_pure.csv')
    video_stat = pd.read_csv('./input/video_features_statistic_pure.csv')

    user_features['user_id'] = user_features['user_id'].astype(str)
    video_basic['video_id'] = video_basic['video_id'].astype(str)
    video_stat['video_id'] = video_stat['video_id'].astype(str)

    user_features.rename(columns={c: f'u_{c}' for c in user_features.columns if c != 'user_id'}, inplace=True)
    video_features = video_basic.merge(video_stat, on='video_id', how='left', suffixes=('_basic', '_stat'))
    video_features['video_id'] = video_features['video_id'].astype(str)
    video_features.rename(columns={c: f'v_{c}' for c in video_features.columns if c != 'video_id'}, inplace=True)

    train_df = train_log.merge(user_features, on='user_id', how='left').merge(video_features, on='video_id', how='left')
    val_df = val_log.merge(user_features, on='user_id', how='left').merge(video_features, on='video_id', how='left')

    user_feat_cols = [c for c in user_features.columns if c != 'user_id']
    video_feat_cols = [c for c in video_features.columns if c != 'video_id']
    all_feat_cols = user_feat_cols + video_feat_cols + ['hour', 'weekday', 'day_index']

    combined = pd.concat([train_df[all_feat_cols], val_df[all_feat_cols]], axis=0)
    categorical_cols = [c for c in all_feat_cols if combined[c].dtype == 'object']
    final_feat_cols = []

    for col in all_feat_cols:
        if col in categorical_cols:
            le = LabelEncoder()
            le.fit(combined[col].astype(str))
            enc_name = col + '_enc'
            train_df[enc_name] = le.transform(train_df[col].astype(str))
            val_df[enc_name] = le.transform(val_df[col].astype(str))
            final_feat_cols.append(enc_name)
        else:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0).astype(np.float32)
            val_df[col] = pd.to_numeric(val_df[col], errors='coerce').fillna(0).astype(np.float32)
            final_feat_cols.append(col)

    train_feat = train_df[final_feat_cols].values.astype(np.float32)
    val_feat = val_df[final_feat_cols].values.astype(np.float32)

    scaler = StandardScaler()
    train_feat = scaler.fit_transform(train_feat).astype(np.float32)
    val_feat = scaler.transform(val_feat).astype(np.float32)

    feat_dim = train_feat.shape[1]

    all_user_ids = pd.concat([train_log['user_id'], val_log['user_id']]).unique()
    all_item_ids = pd.concat([train_log['video_id'], val_log['video_id']]).unique()
    user2idx = {u: i for i, u in enumerate(all_user_ids)}
    item2idx = {v: i for i, v in enumerate(all_item_ids)}
    n_users = len(all_user_ids)
    n_items = len(all_item_ids)

    train_user_idx = train_df['user_id'].map(user2idx).values.astype(np.int64)
    train_item_idx = train_df['video_id'].map(item2idx).values.astype(np.int64)
    val_user_idx = val_df['user_id'].map(user2idx).values.astype(np.int64)
    val_item_idx = val_df['video_id'].map(item2idx).values.astype(np.int64)

    train_labels = train_df['long_view'].values.astype(np.float32)
    val_labels = val_df['long_view'].values.astype(np.float32)

    aux_cols = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward']
    for col in aux_cols:
        if col not in train_df.columns:
            train_df[col] = 0
            val_df[col] = 0
    train_aux = train_df[aux_cols].values.astype(np.float32)
    val_aux = val_df[aux_cols].values.astype(np.float32)

    if 'play_time_ms' not in train_df.columns:
        train_df['play_time_ms'] = 0
        val_df['play_time_ms'] = 0
    train_watch = np.log1p(train_df['play_time_ms'].values.astype(np.float32))
    val_watch = np.log1p(val_df['play_time_ms'].values.astype(np.float32))

    train_dataset = TensorDataset(
        torch.from_numpy(train_user_idx),
        torch.from_numpy(train_item_idx),
        torch.from_numpy(train_feat),
        torch.from_numpy(train_labels),
        torch.from_numpy(train_aux),
        torch.from_numpy(train_watch)
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    model = MultiTaskSharedBottom(n_users, n_items, feat_dim=feat_dim, n_aux=len(aux_cols))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()
    mse = nn.MSELoss()

    epochs = 5
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for user, item, feat, label, aux, watch in train_loader:
            user = user.to(device)
            item = item.to(device)
            feat = feat.to(device)
            label = label.to(device)
            aux = aux.to(device)
            watch = watch.to(device)

            optimizer.zero_grad()
            main_pred, aux_preds, watch_pred = model(user, item, feat)

            loss_main = bce(main_pred, label)
            loss_aux = sum(bce(aux_pred, aux[:, i]) for i, aux_pred in enumerate(aux_preds))
            loss_watch = mse(watch_pred, watch)
            loss = loss_main + loss_aux + 0.5 * loss_watch

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(user)

        print(f'Epoch {epoch + 1} average loss: {total_loss / len(train_loader.dataset):.4f}')

    model.eval()
    val_preds = []
    with torch.no_grad():
        for start in range(0, len(val_user_idx), 1024):
            end = min(start + 1024, len(val_user_idx))
            u = torch.from_numpy(val_user_idx[start:end]).to(device)
            i = torch.from_numpy(val_item_idx[start:end]).to(device)
            f = torch.from_numpy(val_feat[start:end]).to(device)
            main_pred, _, _ = model(u, i, f)
            val_preds.append(main_pred.cpu().numpy())

    val_predictions = np.concatenate(val_preds)
    val_user_ids = val_df['user_id'].tolist()
    val_labels_list = val_df['long_view'].tolist()
    val_predictions_list = val_predictions.tolist()

    val_res = evaluate(val_user_ids, val_labels_list, val_predictions_list)
    print(f'Final Validation Performance: {val_res["primary"]:.4f}')


if __name__ == '__main__':
    main()