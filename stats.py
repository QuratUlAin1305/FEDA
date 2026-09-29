"""Stage 5: Table II rows (mean +/- s.d. over folds) and the FEDA-vs-ESPH test
over 25 paired folds (5 partitions x 5 folds), corrected resampled t-test
(Nadeau & Bengio, 2003). The corrected test is used because CV folds share training data,
which makes uncorrected tests (paired t, Wilcoxon) over-confident."""
import json, numpy as np
from scipy import stats
d = json.load(open('results/results.json'))
print(f"{'configuration':22s} {'R2 mean':>8s} {'s.d.':>7s} {'RMSE':>7s}")
for k, r in d.items():
    if k.endswith('|rand'):
        a = np.array(r['r2'])
        print(f"{k[:-5]:22s} {a.mean():8.4f} {a.std(ddof=1):7.4f} {r['mean_rmse']:7.2f}")
reps = ['rand'] + [f'rep{i}' for i in range(1, 5)]
for L in ('xgb', 'ann'):
    f = np.concatenate([d[f'FEDA|{L}|{r}']['r2'] for r in reps])
    e = np.concatenate([d[f'ESPH|{L}|{r}']['r2'] for r in reps])
    diff = f - e; J = len(diff)
    t = diff.mean() / np.sqrt((1 / J + 0.25) * diff.var(ddof=1))
    p = 2 * stats.t.sf(abs(t), J - 1)
    print(f"\n{L}: FEDA {f.mean():.4f}  ESPH {e.mean():.4f}  mean diff {diff.mean():+.4f}  "
          f"FEDA wins {int((diff > 0).sum())}/{J}  corrected t = {t:.2f}, p = {p:.4f}")
