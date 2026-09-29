"""Stage 1: parse SMILES, embed a single low-energy conformer, cache graph + cloud."""
import sys, pickle, numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

SEED = 0xFEDA

def build(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    molH = Chem.AddHs(mol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = SEED
    ps.useRandomCoords = False
    if AllChem.EmbedMolecule(molH, ps) != 0:
        ps.useRandomCoords = True
        if AllChem.EmbedMolecule(molH, ps) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(molH, maxIters=200)
    except Exception:
        pass
    heavy = Chem.RemoveHs(molH)
    conf = heavy.GetConformer()
    n = heavy.GetNumAtoms()
    X = np.array([list(conf.GetAtomPosition(i)) for i in range(n)], dtype=np.float64)
    Z = np.array([a.GetAtomicNum() for a in heavy.GetAtoms()], dtype=np.int32)
    sym = [a.GetSymbol() for a in heavy.GetAtoms()]
    arom = np.array([a.GetIsAromatic() for a in heavy.GetAtoms()], dtype=bool)
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in heavy.GetBonds()]
    barom = [b.GetIsAromatic() for b in heavy.GetBonds()]
    return dict(n=n, X=X, Z=Z, sym=sym, arom=arom, edges=edges, barom=barom)


def main():
    import openpyxl
    wb = openpyxl.load_workbook('BoilingData.xlsx')
    ws = wb[wb.sheetnames[0]]
    rows = [(r[0], r[1]) for r in ws.iter_rows(min_row=2, values_only=True)
            if r[0] is not None and r[1] is not None]
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    out = []
    for k in range(lo, min(hi, len(rows))):
        smi, bp = rows[k]
        try:
            d = build(smi)
        except Exception:
            d = None
        if d is not None:
            d['smiles'] = smi
            d['bp'] = float(bp)
            d['idx'] = k
            out.append(d)
        if (k - lo) % 200 == 0:
            print(f'{lo}: {k-lo}/{hi-lo}', flush=True)
    with open(f'conf_{lo}.pkl', 'wb') as f:
        pickle.dump(out, f)
    print(f'{lo}: done {len(out)}', flush=True)

if __name__ == '__main__':
    main()
