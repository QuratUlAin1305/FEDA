"""MACCS keys on the running example: three keys highlighted on the molecule and
the full 167-bit vector with the same keys marked."""
import io, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.MACCSkeys import smartsPatts
from palette import *
apply_style()

SMI = "CCN1C(=O)c2ccccc2C1=O"
m = Chem.MolFromSmiles(SMI); AllChem.Compute2DCoords(m)
on = set(MACCSkeys.GenMACCSKeys(m).GetOnBits())
hex2rgb = lambda h: tuple(int(h[i:i+2], 16)/255 for i in (1, 3, 5))
KEYS = [(163, "six-membered ring", TEAL), (154, "C=O", AMBER), (114, r"CH$_3$–CH$_2$–*", RUST)]

hl_atoms, hl_bonds, ac, bc = [], [], {}, {}
for key, _, col in KEYS:
    patt = Chem.MolFromSmarts(smartsPatts[key][0])
    for match in m.GetSubstructMatches(patt)[:2 if key == 154 else 1]:
        for a in match:
            hl_atoms.append(a); ac[a] = hex2rgb(col)
        for b in m.GetBonds():
            if b.GetBeginAtomIdx() in match and b.GetEndAtomIdx() in match:
                hl_bonds.append(b.GetIdx()); bc[b.GetIdx()] = hex2rgb(col)
d = rdMolDraw2D.MolDraw2DCairo(900, 620)
o = d.drawOptions(); o.bondLineWidth = 3; o.highlightRadius = 0.42; o.fixedFontSize = 34
o.setHighlightColour(hex2rgb(TEAL_LT)); o.clearBackground = True
d.DrawMolecule(m, highlightAtoms=list(set(hl_atoms)), highlightAtomColors=ac,
               highlightBonds=list(set(hl_bonds)), highlightBondColors=bc)
d.FinishDrawing(); img = Image.open(io.BytesIO(d.GetDrawingText()))

fig = plt.figure(figsize=(3.45, 1.72))
ax1 = fig.add_axes([0.0, 0.0, 0.44, 0.9]); ax1.imshow(img); ax1.axis("off")
ax1.set_title("(a) keys matched on the molecule", fontsize=8.2, pad=2, color=SLATE_DK)
ax2 = fig.add_axes([0.47, 0.34, 0.52, 0.58])
R, C = 8, 21
kc = {k: c for k, _, c in KEYS}
for b in range(R*C):
    r, c = divmod(b, C)
    if b >= 167:
        continue
    face = kc.get(b, SLATE if b in on else PAPER)
    ax2.add_patch(Rectangle((c, R-1-r), 0.9, 0.9, facecolor=face,
                            edgecolor=SLATE_LT if face == PAPER else face, lw=0.4))
ax2.set_xlim(-0.1, C); ax2.set_ylim(-0.1, R); ax2.set_aspect("equal"); ax2.axis("off")
ax2.set_title(f"(b) 167-bit MACCS vector, {len(on)} bits set", fontsize=8.2, pad=2, color=SLATE_DK)
for i, (key, lab, col) in enumerate(KEYS):
    y = 0.22 - i*0.10
    fig.patches.append(Rectangle((0.49, y), 0.022, 0.06, transform=fig.transFigure, facecolor=col, edgecolor=col))
    fig.text(0.525, y+0.03, f"key {key}: {lab}", fontsize=7.2, va="center", color=SLATE_DK)
fig.savefig("c7_maccs.png", dpi=400, facecolor=PAPER)
print("maccs ok", sorted(on))
