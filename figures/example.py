"""Running example molecule used in the methodology figures."""
import sys; sys.path.insert(0, "..")
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

SMI  = "CCN1C(=O)c2ccccc2C1=O"
NAME = "2-ethyl-1H-isoindole-1,3(2H)-dione"
SHORT = "N-ethylphthalimide"
SRC = '../BoilingData.xlsx'

def load():
    df = pd.read_excel(SRC)
    idx = int(df.index[df.SMILES == SMI][0])
    bp = float(df.BP.iloc[idx])
    import conformers as C
    pos = C.build(SMI)['X']
    m = Chem.MolFromSmiles(SMI)
    c = Chem.Conformer(m.GetNumAtoms())
    for j, p in enumerate(pos):
        c.SetAtomPosition(j, [float(x) for x in p])
    m.AddConformer(c, assignId=True)
    return m, np.asarray(pos, dtype=float), bp

if __name__ == "__main__":
    m, pos, bp = load()
    print(SHORT, "BP", bp, "heavy atoms", m.GetNumAtoms(), "bonds", m.GetNumBonds())
    print("elements", [a.GetSymbol() for a in m.GetAtoms()])
