import csv
import numpy as np
from evaluate import evaluate

TRAIN_FILE = './input/log_standard_4_08_to_4_21_pure.csv'
VAL_FILE = './input/log_standard_4_22_to_5_08_pure.csv'

USER_COL = 'user_id'
VIDEO_COL = 'video_id'
DATE_COL = 'date'
LABEL_COL = 'long_view'


def load_rows(path, start, end):
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for rec in reader:
            d = rec[DATE_COL].strip()
            if d < start or d > end:
                continue
            rows.append((
                rec[USER_COL].strip(),
                rec[VIDEO_COL].strip(),
                int(float(rec[LABEL_COL]))
            ))
    return rows


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


def build_data(rows, user2idx, video2idx):
    X = np.empty((len(rows), 2), dtype=np.int32)
    y = np.empty(len(rows), dtype=np.float32)
    for i, (u, v, label) in enumerate(rows):
        X[i, 0] = user2idx[u]
        X[i, 1] = video2idx[v]
        y[i] = label
    return X, y


def main():
    train_rows = load_rows(TRAIN_FILE, '20220408', '20220421')
    val_rows = load_rows(VAL_FILE, '20220422', '20220428')

    if len(train_rows) > 30000:
        rng = np.random.default_rng(0)
        chosen = rng.choice(len(train_rows), size=30000, replace=False)
        train_rows = [train_rows[i] for i in chosen]

    users = set()
    videos = set()
    for rows in (train_rows, val_rows):
        for u, v, _ in rows:
            users.add(u)
            videos.add(v)

    user_list = sorted(users)
    video_list = sorted(videos)
    user2idx = {u: i for i, u in enumerate(user_list)}
    video2idx = {v: i + len(user_list) for i, v in enumerate(video_list)}
    dim = len(user_list) + len(video_list)

    Xtr, ytr = build_data(train_rows, user2idx, video2idx)
    Xva, yva = build_data(val_rows, user2idx, video2idx)
    uva = [r[0] for r in val_rows]

    model = FM(dim, k=16, lr=0.001, l2=1e-6, seed=0)
    rng = np.random.default_rng(0)

    best = -1.0
    bad = 0
    best_state = None
    epochs = 40
    bs = 8192
    patience = 4

    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), bs):
            batch_idx = idx[i:i + bs]
            model.step(Xtr[batch_idx], ytr[batch_idx])

        pred = model.predict(Xva)
        va = evaluate(uva, yva.tolist(), pred.tolist())

        if va['primary'] > best + 1e-5:
            best = va['primary']
            bad = 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= patience:
                break

    model.V, model.W, model.b = best_state
    final_pred = model.predict(Xva)
    final_res = evaluate(uva, yva.tolist(), final_pred.tolist())

    print(f'Final Validation Performance: {final_res["primary"]:.4f}')


if __name__ == '__main__':
    main()