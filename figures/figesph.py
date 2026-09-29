"""ESPH figure: Vietoris-Rips filtration and barcode, one consistent convention."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, gudhi
from matplotlib.patches import Circle
from itertools import combinations
from palette import *
import conformers as C
apply_style()
plt.rcParams.update({"font.size": 8.4})

d = C.build('CCN1C(=O)c2ccccc2C1=O')          # N-ethylphthalimide
S = [i for i, s in enumerate(d['sym']) if s == 'C']
Y = d['X'][S]
Dm = np.sqrt(((Y[:, None] - Y[None]) ** 2).sum(-1))

st = gudhi.RipsComplex(points=Y, max_edge_length=12.0).create_simplex_tree(max_dimension=3)
st.compute_persistence(homology_coeff_field=2, persistence_dim_max=False)
bars = {k: sorted([p for p in st.persistence_intervals_in_dimension(k)
                   if np.isfinite(p[1])], key=lambda p: p[0] - p[1])
        for k in (0, 1, 2)}

Yc = Y - Y.mean(0)
_, _, Vt = np.linalg.svd(Yc)
P = Yc @ Vt[:2].T

SCALES = [1.30, 1.75, 2.60]
COL = {0: TEAL, 1: RUST, 2: AMBER_DK}

fig = plt.figure(figsize=(7.4, 3.30))
gs = fig.add_gridspec(2, 3, height_ratios=[0.80, 1.30], hspace=0.24, wspace=0.06,
                      left=0.055, right=0.985, top=0.878, bottom=0.092)

# ------------------------------------------------- (a) three filtration snapshots
lim = np.abs(P).max() + 1.5
for c, r in enumerate(SCALES):
    ax = fig.add_subplot(gs[0, c]); ax.set_aspect('equal'); ax.axis('off')
    for i, j in combinations(range(len(P)), 2):
        if Dm[i, j] <= r:
            ax.plot(*zip(P[i], P[j]), color=TEAL, lw=1.25, zorder=2, alpha=0.9)
    for p in P:                                   # discs of radius r/2:
        ax.add_patch(Circle(p, r / 2, facecolor=TEAL_PALE, edgecolor='none',
                            alpha=0.75, zorder=1))   # they touch exactly when d <= r
    ax.scatter(P[:, 0], P[:, 1], s=17, color=SLATE_DK, zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim * 0.50, lim * 0.50)
    ax.set_title(f'$r = {r:.2f}$ Å', fontsize=8.6, color=SLATE_DK, pad=3)
fig.text(0.52, 0.978, '(a) Vietoris–Rips filtration of the carbon subcloud '
                      '$X_{\\mathrm{C}}$:  a simplex enters when its vertices are '
                      'pairwise within $r$', ha='center', fontsize=8.6, color=SLATE_DK)
fig.text(0.52, 0.938, 'discs drawn at radius $r/2$, so two discs meet exactly when '
                      '$\\|x_i-x_j\\| \\leq r$',
         ha='center', fontsize=7.2, color=SLATE_MD)

# ------------------------------------------------- (b) barcode
ax = fig.add_subplot(gs[1, :])
y = 0
rows = []
for k in (2, 1, 0):
    for b, dd in bars[k]:
        ax.plot([b, dd], [y, y], color=COL[k], lw=3.0, solid_capstyle='butt', zorder=3)
        rows.append((k, y, b, dd)); y += 1
    y += 0.9

for r in SCALES:
    ax.axvline(r, color=AMBER_DK, ls=(0, (3.5, 2.5)), lw=0.95, zorder=1, alpha=0.85)
    ax.text(r, -1.20, f'{r:.2f}', ha='center', va='bottom', fontsize=6.9,
            color=AMBER_DK)

for k, lbl in ((0, '$H_0$'), (1, '$H_1$'), (2, '$H_2$')):
    ys = [yy for kk, yy, _, _ in rows if kk == k]
    ax.text(-0.62, np.mean(ys), lbl, ha='right', va='center', fontsize=9.4,
            color=COL[k], fontweight='bold')

# every H0 class is born at r = 0 -- the point that makes its mean birth vacuous
h0y = [yy for kk, yy, _, _ in rows if kk == 0]
h0top, h0bot = max(h0y), min(h0y)
ax.plot([0, 0], [h0bot - 0.45, h0top + 0.45], color=TEAL, lw=1.8, zorder=5)
ax.text(-0.14, (h0top + h0bot) / 2, 'all $H_0$ births at $r=0$', rotation=90,
        ha='right', va='center', fontsize=7.2, color=TEAL)
ax.text(2.72, h0top + 0.05, 'so the degree-0 mean birth is identically $0$,\nand the '
        'degree-0 mean death equals the mean lifetime', fontsize=7.2, color=TEAL,
        va='center', ha='left')

# one H1 bar, labelled with b, d and L
k1, y1, b1, d1 = [row for row in rows if row[0] == 1][0]
ax.annotate('', xy=(b1, y1 + 0.62), xytext=(d1, y1 + 0.62),
            arrowprops=dict(arrowstyle='<->', color=SLATE, lw=0.85))
ax.text((b1 + d1) / 2, y1 + 0.80, f'$L=d-b={d1-b1:.2f}$', ha='center', va='bottom',
        fontsize=7.3, color=SLATE_DK)
for v, t in ((b1, '$b$'), (d1, '$d$')):
    ax.text(v, y1 - 0.62, t, ha='center', va='top', fontsize=7.6, color=SLATE_DK)

ax.set_xlim(-0.95, 4.35); ax.set_ylim(-1.55, y + 0.35)
ax.set_yticks([]); ax.set_xlabel('filtration scale  $r$  (Å)', fontsize=8.6)
ax.grid(axis='x', color=SLATE_LT, lw=0.5, alpha=0.8, zorder=0); ax.set_axisbelow(True)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.set_title('(b) the resulting barcode, truncated at the $12$ Å cut-off '
             '(the one essential $H_0$ class is not shown)',
             fontsize=8.6, color=SLATE_DK, pad=4)

fig.savefig('esph_fig.png', dpi=400, facecolor=PAPER, bbox_inches='tight')
print('bars:', {k: len(v) for k, v in bars.items()})
