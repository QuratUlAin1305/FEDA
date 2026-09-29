import numpy as np, time
from common import *
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
RDLogger.DisableLog('rdApp.*')
import pandas as pd
idx, y, Xb = load_feats()
smi = pd.read_excel('BoilingData.xlsx').SMILES.values[idx]
mols = [Chem.MolFromSmiles(s) for s in smi]
Xb['MACCS'] = np.array([np.array(list(MACCSkeys.GenMACCSKeys(m)), float) for m in mols])
F = Xb
F['FEDA-FG'] = np.hstack([F['ESPH'], F['ADM'], F['AM']])
F['FEDA-ESPH'] = np.hstack([F['FG'], F['ADM'], F['AM']])
F['FEDA-ADM'] = np.hstack([F['FG'], F['ESPH'], F['AM']])
F['FEDA-AM'] = np.hstack([F['FG'], F['ESPH'], F['ADM']])
F['ESPH+FG'] = np.hstack([F['ESPH'], F['FG']])
S = Store('results/results.json'); folds = random_folds(len(y))
order = ['FG','AM','ADM','ESPH','FEDA','MACCS','ESPH+FG','FEDA-FG','FEDA-ESPH','FEDA-ADM','FEDA-AM']
for name in order:
    for L in ('xgb','ann'):
        k = f'{name}|{L}|rand'
        if S.has(k): continue
        t = time.time(); r = run(F[name], y, folds, L); r['dim'] = int(F[name].shape[1]); r['sec'] = time.time()-t
        S.put(k, r); print(k, r['dim'], round(r['mean_r2'],4), round(r['mean_rmse'],2), int(r['sec']), flush=True)
# repeated CV, 4 more partitions, FEDA vs ESPH
for rep, sd in enumerate([11, 22, 33, 44], 1):
    fr = random_folds(len(y), sd)
    for name in ('ESPH','FEDA'):
        for L in ('xgb','ann'):
            k = f'{name}|{L}|rep{rep}'
            if S.has(k): continue
            r = run(F[name], y, fr, L); S.put(k, r); print(k, round(r['mean_r2'],4), flush=True)
print('DONE', flush=True)
