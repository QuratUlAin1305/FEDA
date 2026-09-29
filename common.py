import json, os, time, numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
SEED = 20260902
BLOCKS = ['FG', 'AM', 'ADM', 'ESPH']

def load_feats(prefix='feat_'):
    parts = [np.load(f'{prefix}{lo}.npz') for lo in (0, 1400, 2800, 4200)]
    idx = np.concatenate([p['idx'] for p in parts]); order = np.argsort(idx)
    y = np.concatenate([p['y'] for p in parts])[order]
    Xb = {b: np.concatenate([p[b] for p in parts])[order] for b in BLOCKS}
    Xb['FEDA'] = np.concatenate([Xb['FG'], Xb['ESPH'], Xb['ADM'], Xb['AM']], axis=1)
    return idx[order], y, Xb

def prep(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-9] = 1.0
    return np.clip((Xtr-mu)/sd, -5, 5), np.clip((Xte-mu)/sd, -5, 5)

def fit_predict(Xtr, ytr, Xte, learner, seed=SEED):
    if learner == 'xgb':
        m = XGBRegressor(n_estimators=1200, max_depth=8, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=1,
                         tree_method='hist', n_jobs=2, random_state=SEED, verbosity=0)
        m.fit(Xtr, ytr); return m.predict(Xte)
    Xtr, Xte = prep(Xtr, Xte); ym, ys = ytr.mean(), ytr.std()
    m = MLPRegressor(hidden_layer_sizes=(1024, 512, 256), solver='adam', activation='relu',
                     alpha=1e-4, batch_size=128, learning_rate_init=1e-3, max_iter=400,
                     early_stopping=True, n_iter_no_change=20, validation_fraction=0.1,
                     random_state=SEED)
    m.fit(Xtr, (ytr-ym)/ys); return m.predict(Xte)*ys + ym

def run(X, y, folds, learner):
    r2s, rmses = [], []
    for tr, te in folds:
        p = fit_predict(X[tr], y[tr], X[te], learner)
        r2s.append(float(r2_score(y[te], p))); rmses.append(float(np.sqrt(mean_squared_error(y[te], p))))
    return dict(r2=r2s, rmse=rmses, mean_r2=float(np.mean(r2s)), mean_rmse=float(np.mean(rmses)))

def random_folds(n, seed=SEED):
    return list(KFold(n_splits=5, shuffle=True, random_state=seed).split(np.arange(n)))

class Store:
    def __init__(self, path): self.path = path; self.d = json.load(open(path)) if os.path.exists(path) else {}
    def has(self, k): return k in self.d
    def put(self, k, v): self.d[k] = v; json.dump(self.d, open(self.path, 'w'), indent=1)
