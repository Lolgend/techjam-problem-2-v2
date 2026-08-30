import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import cross_val_predict, KFold
from collections import deque
from evaluate import evaluate


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr = lr
        self.l2 = l2
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)

        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V
        gW += self.l2 * self.W

        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in (
            (self.V, gV, self.mV, self.vV),
            (self.W, gW, self.mW, self.vW),
        ):
            M *= b1
            M += (1 - b1) * G
            Vv *= b2
            Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)

        self.b -= self.lr * g.sum()
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def predict(self, X, bs=200000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


def bpr_loss(scores, labels, user_ids):
    device = scores.device
    loss = torch.tensor(0.0, device=device)
    total = 0
    for u in torch.unique(user_ids):
        idx = (user_ids == u).nonzero(as_tuple=False).squeeze(1)
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] < 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        diff = scores[pos][:, None] - scores[neg][None, :]
        loss = loss + (-F.logsigmoid(diff)).sum()
        total += pos.numel() * neg.numel()
    if total > 0:
        return loss / total
    return loss


def main():
    set_seed(42)

    # Limit input size to avoid extremely long execution.
    TRAIN_READ_NROWS = 250000
    LATER_READ_NROWS = 250000
    MAX_TRAIN_ROWS = 100000
    MAX_VAL_ROWS = 60000
    EPOCHS = 2
    DIN_EPOCHS = 1
    FM_EPOCHS = 10
    BATCH_SIZE = 256
    MAX_HIST = 20

    train_log = pd.read_csv('./input/log_standard_4_08_to_4_21_pure.csv', nrows=TRAIN_READ_NROWS)
    later_log = pd.read_csv('./input/log_standard_4_22_to_5_08_pure.csv', nrows=LATER_READ_NROWS)

    date_col = 'date' if 'date' in train_log.columns else ('day' if 'day' in train_log.columns else None)
    if date_col is None:
        raise ValueError('No date column found.')

    train_log['date_int'] = train_log[date_col].apply(parse_date_to_int)
    later_log['date_int'] = later_log[date_col].apply(parse_date_to_int)

    train_log['date_str'] = train_log['date_int'].apply(
        lambda x: str(int(x))[:8] if pd.notna(x) else ''
    )
    later_log['date_str'] = later_log['date_int'].apply(
        lambda x: str(int(x))[:8] if pd.notna(x) else ''
    )

    train_log = train_log[
        (train_log['date_int'] >= 20220408) & (train_log['date_int'] <= 20220421)
    ].reset_index(drop=True)

    val_log = later_log[
        (later_log['date_int'] >= 20220422) & (later_log['date_int'] <= 20220428)
    ].reset_index(drop=True)

    if len(train_log) > MAX_TRAIN_ROWS:
        train_log = train_log.sample(n=MAX_TRAIN_ROWS, random_state=42).reset_index(drop=True)

    if len(val_log) > MAX_VAL_ROWS:
        val_log = val_log.sample(n=MAX_VAL_ROWS, random_state=42).reset_index(drop=True)

    for df in [train_log, val_log]:
        df['user_id'] = df['user_id'].astype(str)
        df['video_id'] = df['video_id'].astype(str)

        for c in ['hour', 'minute', 'second']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(np.float32)

        if 'hour' not in df.columns:
            df['hour'] = np.float32(0)

        dt = pd.to_datetime(df['date_int'].astype(str), format='%Y%m%d', errors='coerce')
        df['weekday'] = dt.dt.weekday.fillna(0).astype(int)
        df['day_index'] = df['date_int'] - 20220408

    sort_cols = ['user_id', 'date_str']
    for c in ['hour', 'minute', 'second']:
        if c in train_log.columns:
            sort_cols.append(c)

    train_log = train_log.sort_values(sort_cols).reset_index(drop=True)

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

    for df in [train_df, val_df]:
        df['long_view'] = pd.to_numeric(df['long_view'], errors='coerce').fillna(0).astype(np.float32)

    user_feat_cols = [c for c in user_features.columns if c != 'user_id']
    video_feat_cols = [c for c in video_features.columns if c != 'video_id']
    time_feat_cols = ['hour', 'weekday', 'day_index']
    for c in ['minute', 'second']:
        if c in train_df.columns:
            time_feat_cols.append(c)
    all_feat_cols = user_feat_cols + video_feat_cols + time_feat_cols

    categorical_cols = [c for c in all_feat_cols if train_df[c].dtype == 'object']
    final_feat_cols = []

    for col in all_feat_cols:
        if col in categorical_cols:
            le = LabelEncoder()
            le.fit(train_df[col].astype(str))
            enc_name = col + '_enc'
            train_df[enc_name] = le.transform(train_df[col].astype(str))

            class_to_code = {cls: i for i, cls in enumerate(le.classes_)}
            val_df[enc_name] = val_df[col].astype(str).map(lambda x: class_to_code.get(x, -1)).astype(np.int64)
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

    train_users = train_log['user_id'].unique()
    train_items = train_log['video_id'].unique()

    user2idx = {u: i + 1 for i, u in enumerate(train_users)}
    item2idx = {v: i + 1 for i, v in enumerate(train_items)}
    n_users = len(train_users) + 1
    n_items = len(train_items) + 1

    train_user_idx = train_df['user_id'].map(user2idx).fillna(0).astype(np.int64).values
    train_item_idx = train_df['video_id'].map(item2idx).fillna(0).astype(np.int64).values
    val_user_idx = val_df['user_id'].map(user2idx).fillna(0).astype(np.int64).values
    val_item_idx = val_df['video_id'].map(item2idx).fillna(0).astype(np.int64).values

    train_labels = train_df['long_view'].values.astype(np.float32)
    val_labels = val_df['long_view'].values.astype(np.float32)

    aux_cols = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward']
    for col in aux_cols:
        if col not in train_df.columns or col not in val_df.columns:
            train_df[col] = 0
            val_df[col] = 0
        else:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0).astype(np.float32)
            val_df[col] = pd.to_numeric(val_df[col], errors='coerce').fillna(0).astype(np.float32)

    train_aux = train_df[aux_cols].values.astype(np.float32)
    val_aux = val_df[aux_cols].values.astype(np.float32)

    if 'play_time_ms' not in train_df.columns:
        train_df['play_time_ms'] = 0
    if 'play_time_ms' not in val_df.columns:
        val_df['play_time_ms'] = 0

    train_df['play_time_ms'] = pd.to_numeric(train_df['play_time_ms'], errors='coerce').fillna(0)
    val_df['play_time_ms'] = pd.to_numeric(val_df['play_time_ms'], errors='coerce').fillna(0)

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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- MultiTaskSharedBottom ----
    model = MultiTaskSharedBottom(n_users, n_items, feat_dim=feat_dim, n_aux=len(aux_cols))
    model.to(device)

    model.user_emb.weight.data[0] = 0.0
    model.item_emb.weight.data[0] = 0.0

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()
    mse = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
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
            loss_rank = bpr_loss(main_pred, label, user)
            loss = loss_main + loss_aux + 0.5 * loss_watch + 0.1 * loss_rank

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(user)

    model.eval()
    base_val_preds = []
    with torch.no_grad():
        for start in range(0, len(val_user_idx), 1024):
            end = min(start + 1024, len(val_user_idx))
            u = torch.from_numpy(val_user_idx[start:end]).to(device)
            i = torch.from_numpy(val_item_idx[start:end]).to(device)
            f = torch.from_numpy(val_feat[start:end]).to(device)
            main_pred, _, _ = model(u, i, f)
            base_val_preds.append(main_pred.cpu().numpy())

    base_val_predictions = np.concatenate(base_val_preds)

    # ---- DIN ----
    d_user2idx = {u: i + 1 for i, u in enumerate(train_users)}
    d_item2idx = {v: i + 1 for i, v in enumerate(train_items)}
    n_users_d = len(train_users) + 1
    n_items_d = len(train_items) + 1

    train_df['u_idx_d'] = train_df['user_id'].map(d_user2idx).fillna(0).astype(np.int64)
    train_df['i_idx_d'] = train_df['video_id'].map(d_item2idx).fillna(0).astype(np.int64)
    val_df['u_idx_d'] = val_df['user_id'].map(d_user2idx).fillna(0).astype(np.int64)
    val_df['i_idx_d'] = val_df['video_id'].map(d_item2idx).fillna(0).astype(np.int64)

    set_seed(42)

    # Vectorized history construction instead of per-row Python loops.
    max_hist = MAX_HIST
    combined_df = pd.concat(
        [train_df[['u_idx_d', 'i_idx_d']], val_df[['u_idx_d', 'i_idx_d']]],
        ignore_index=True
    )
    combined_hist = np.zeros((len(combined_df), max_hist), dtype=np.int64)
    combined_hist_mask = np.zeros((len(combined_df), max_hist), dtype=np.bool_)

    for lag in range(1, max_hist + 1):
        shifted = combined_df.groupby('u_idx_d')['i_idx_d'].shift(lag)
        col = max_hist - lag
        combined_hist[:, col] = shifted.fillna(0).to_numpy(dtype=np.int64)
        combined_hist_mask[:, col] = shifted.notna().to_numpy(dtype=np.bool_)

    train_hist = combined_hist[:len(train_df)]
    train_hist_mask = combined_hist_mask[:len(train_df)]
    val_hist = combined_hist[len(train_df):]
    val_hist_mask = combined_hist_mask[len(train_df):]

    train_u_d = train_df['u_idx_d'].to_numpy(dtype=np.int64)
    train_i_d = train_df['i_idx_d'].to_numpy(dtype=np.int64)
    train_y_d = train_df['long_view'].to_numpy(dtype=np.float32)

    val_u_d = val_df['u_idx_d'].to_numpy(dtype=np.int64)
    val_i_d = val_df['i_idx_d'].to_numpy(dtype=np.int64)

    model_d = DIN(
        n_users=n_users_d,
        n_items=n_items_d,
        embed_dim=16,
        feat_dim=feat_dim,
        max_hist=max_hist
    ).to(device)

    with torch.no_grad():
        model_d.user_emb.weight.data[0] = 0.0
        model_d.item_emb.weight.data[0] = 0.0
        model_d.hist_emb.weight.data[0] = 0.0

    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    model_d.train()
    for epoch in range(DIN_EPOCHS):
        perm = np.random.permutation(len(train_u_d))
        for i in range(0, len(train_u_d), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]

            user = torch.tensor(train_u_d[idx], dtype=torch.long, device=device)
            item = torch.tensor(train_i_d[idx], dtype=torch.long, device=device)
            hist = torch.tensor(train_hist[idx], dtype=torch.long, device=device)
            hist_mask = torch.tensor(train_hist_mask[idx], dtype=torch.bool, device=device)
            feat = torch.tensor(train_feat[idx], dtype=torch.float32, device=device)
            y = torch.tensor(train_y_d[idx], dtype=torch.float32, device=device)

            logits = model_d(user, item, hist, hist_mask, feat)
            loss_bce = loss_fn(logits, y)
            loss_rank = bpr_loss(logits, y, user)
            loss = loss_bce + 0.1 * loss_rank

            optimizer_d.zero_grad()
            loss.backward()
            optimizer_d.step()

    model_d.eval()
    din_val_preds = []

    with torch.no_grad():
        for i in range(0, len(val_u_d), BATCH_SIZE):
            user = torch.tensor(val_u_d[i:i + BATCH_SIZE], dtype=torch.long, device=device)
            item = torch.tensor(val_i_d[i:i + BATCH_SIZE], dtype=torch.long, device=device)
            hist = torch.tensor(val_hist[i:i + BATCH_SIZE], dtype=torch.long, device=device)
            hist_mask = torch.tensor(val_hist_mask[i:i + BATCH_SIZE], dtype=torch.bool, device=device)
            feat = torch.tensor(val_feat[i:i + BATCH_SIZE], dtype=torch.float32, device=device)

            logits = model_d(user, item, hist, hist_mask, feat)
            din_val_preds.append(torch.sigmoid(logits).cpu().numpy())

    din_val_predictions = np.concatenate(din_val_preds)

    # ---- FM ----
    train_fm_user = train_df['user_id'].map(user2idx).fillna(0).astype(np.int32).values
    train_fm_video = train_df['video_id'].map(item2idx).fillna(0).astype(np.int32).values + n_users
    Xtr_fm = np.stack([train_fm_user, train_fm_video], axis=1)
    ytr_fm = train_df['long_view'].values.astype(np.float32)

    val_fm_user = val_df['user_id'].map(user2idx).fillna(0).astype(np.int32).values
    val_fm_video = val_df['video_id'].map(item2idx).fillna(0).astype(np.int32).values + n_users
    Xva_fm = np.stack([val_fm_user, val_fm_video], axis=1)

    dim_fm = n_users + n_items
    model_fm = FM(dim_fm, k=16, lr=0.001, l2=1e-6, seed=0)

    model_fm.V[0] = 0.0
    model_fm.W[0] = 0.0
    model_fm.V[n_users] = 0.0
    model_fm.W[n_users] = 0.0

    rng_fm = np.random.default_rng(0)

    bs_fm = 8192
    for ep in range(1, FM_EPOCHS + 1):
        idx_fm = rng_fm.permutation(len(ytr_fm))
        for i in range(0, len(idx_fm), bs_fm):
            batch_idx = idx_fm[i:i + bs_fm]
            model_fm.step(Xtr_fm[batch_idx], ytr_fm[batch_idx])

    fm_val_logits = model_fm.predict(Xva_fm)
    fm_val_predictions = sigmoid(fm_val_logits)

    # ---- Blend ----
    y_val_arr = np.asarray(val_labels).ravel().astype(np.float64)

    val_preds = [
        np.asarray(base_val_predictions, dtype=np.float64).ravel(),
        np.asarray(din_val_predictions, dtype=np.float64).ravel(),
        np.asarray(fm_val_predictions, dtype=np.float64).ravel(),
    ]

    n_models = len(val_preds)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    calibrated_val = np.empty((len(y_val_arr), n_models))

    for i, val_pred in enumerate(val_preds):
        X_val = val_pred.reshape(-1, 1)

        iso_cv = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        calibrated_val[:, i] = cross_val_predict(iso_cv, X_val, y_val_arr, cv=cv, method='predict')

    calibrated_val = np.clip(calibrated_val, 0.0, 1.0)

    errors = y_val_arr[:, None] - calibrated_val
    lw = LedoitWolf().fit(errors)
    cov = lw.covariance_

    inv_cov = np.linalg.solve(cov, np.ones(n_models))
    weights = inv_cov / np.sum(inv_cov)

    final_val_predictions = np.clip(calibrated_val.dot(weights), 0.0, 1.0)

    val_user_ids = val_df['user_id'].tolist()
    val_labels_list = val_df['long_view'].astype(int).tolist()

    val_res = evaluate(val_user_ids, val_labels_list, final_val_predictions.tolist())
    print(f'Final Validation Performance: {val_res["primary"]:.4f}')


if __name__ == '__main__':
    main()