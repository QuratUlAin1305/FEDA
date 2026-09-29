"""Stage 3: five-fold CV on one fixed partition, reused for every representation."""
import json, numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

SEED = 20260902
BLOCKS = ['FG', 'AM', 'ADM', 'ESPH']

parts = [np.load(f'feat_{lo}.npz') for lo in (0, 1400, 2800, 4200)]
idx = np.concatenate([p['idx'] for p in parts])
order = np.argsort(idx)
y = np.concatenate([p['y'] for p in parts])[order]
Xb = {b: np.concatenate([p[b] for p in parts])[order] for b in BLOCKS}
Xb['FEDA'] = np.concatenate([Xb['FG'], Xb['ESPH'], Xb['ADM'], Xb['AM']], axis=1)
print('n =', len(y), 'dims =', {k: v.shape[1] for k, v in Xb.items()}, flush=True)

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(kf.split(y))


def prep(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1.0
    a = np.clip((Xtr - mu) / sd, -5, 5)
    b = np.clip((Xte - mu) / sd, -5, 5)
    return a, b


def run(X, learner):
    r2s, rmses = [], []
    for tr, te in folds:
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        if learner == 'xgb':
            m = XGBRegressor(n_estimators=1200, max_depth=8, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, min_child_weight=1,
                             tree_method='hist', n_jobs=2, random_state=SEED,
                             verbosity=0)
            m.fit(Xtr, ytr)
            pred = m.predict(Xte)
        else:
            Xtr, Xte = prep(Xtr, Xte)
            ym, ys = ytr.mean(), ytr.std()
            m = MLPRegressor(hidden_layer_sizes=(1024, 512, 256), solver='adam',
                             activation='relu', alpha=1e-4, batch_size=128,
                             learning_rate_init=1e-3, max_iter=400,
                             early_stopping=True, n_iter_no_change=20,
                             validation_fraction=0.1, random_state=SEED)
            m.fit(Xtr, (ytr - ym) / ys)
            pred = m.predict(Xte) * ys + ym
        r2s.append(r2_score(yte, pred))
        rmses.append(np.sqrt(mean_squared_error(yte, pred)))
    return r2s, rmses


import os
results = json.load(open('results.json')) if os.path.exists('results.json') else {}
for name in BLOCKS + ['FEDA']:
    for learner in ('xgb', 'ann'):
        if f'{name}_{learner}' in results:
            print('skip', name, learner, flush=True); continue
        r2s, rmses = run(Xb[name], learner)
        results[f'{name}_{learner}'] = dict(r2=r2s, rmse=rmses,
                                            mean_r2=float(np.mean(r2s)),
                                            mean_rmse=float(np.mean(rmses)))
        print(f'{name:5s} {learner:4s} d={Xb[name].shape[1]:3d} '
              f'R2={np.mean(r2s):.4f} RMSE={np.mean(rmses):.2f}', flush=True)
        json.dump(results, open('results.json', 'w'), indent=1)

# per-fold FEDA vs best single block, for the "every fold" claim
print('\nper-fold R2:', flush=True)
for k in results:
    print(k, [round(v, 4) for v in results[k]['r2']], flush=True)
print('\ntarget stats: mean %.1f sd %.1f range %.1f'
      % (y.mean(), y.std(), y.max() - y.min()), flush=True)
