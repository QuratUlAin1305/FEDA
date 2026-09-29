"""Column-width versions of the distance- and angle-matrix figures (2-butanone)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, networkx as nx
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.colors import LinearSegmentedColormap
from itertools import combinations
from palette import *
import conformers as C
apply_style()
plt.rcParams.update({"font.size": 7.4})

d = C.build("CC(=O)CC")
X, sym, n, edges = d['X'], d['sym'], d['n'], d['edges']
lab, ci = [], 1
for s in sym:
    lab.append(f'C$_{ci}$') if s == 'C' else lab.append('O')
    if s == 'C':
        ci += 1
G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(edges)
adj = {i: list(G.neighbors(i)) for i in range(n)}
D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
sp = dict(nx.all_pairs_shortest_path_length(G))

Theta = np.zeros((n, n)); central = {}
for k in range(n):
    for i, j in combinations(adj[k], 2):
        v1, v2 = X[i] - X[k], X[j] - X[k]
        t = np.degrees(np.arccos(np.clip(np.dot(v1, v2) /
                                         (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
        Theta[i, j] = Theta[j, i] = t
        central[(min(i, j), max(i, j))] = k

Xc = X - X.mean(0); _, _, Vt = np.linalg.svd(Xc); P = Xc @ Vt[:2].T


def grid(ax, M, mask, cmap, vmin, vmax, fmt, boldfn):
    ax.set_aspect('equal')
    ax.imshow(np.where(mask, M, np.nan), cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=SLATE_LT,
                                       edgecolor='none', zorder=2))
                ax.text(j, i, '0', ha='center', va='center', fontsize=6.9,
                        color=SLATE_DK, zorder=3, fontweight='bold')
            elif mask[i, j]:
                ax.text(j, i, fmt(M[i, j]), ha='center', va='center', fontsize=6.4,
                        color=SLATE_DK, zorder=3,
                        fontweight='bold' if boldfn(i, j) else 'normal')
            else:
                ax.text(j, i, '0', ha='center', va='center', fontsize=6.6,
                        color=SLATE_MD, zorder=3)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(lab, fontsize=6.9); ax.set_yticklabels(lab, fontsize=6.9)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which='minor', color=PAPER, linewidth=1.3)
    ax.tick_params(which='both', length=0)
    for sp_ in ax.spines.values():
        sp_.set_color(SLATE_MD); sp_.set_linewidth(0.7)


def nodes(ax):
    for i in range(n):
        edge = SLATE_DK if sym[i] == 'C' else AMBER_DK
        ax.scatter(*P[i], s=210, color=PAPER, edgecolor=edge, linewidth=1.3, zorder=4)
        ax.text(*P[i], lab[i], ha='center', va='center', fontsize=6.8, color=edge, zorder=5)


# ---------------------------------------------------------------- distance
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.45, 2.52),
                             gridspec_kw=dict(height_ratios=[1.18, 1.0], hspace=0.15))
fig.subplots_adjust(left=0.13, right=0.985, top=0.935, bottom=0.055)
a1.set_aspect('equal'); a1.axis('off')
bonded = {tuple(sorted(e)) for e in edges}
for i, j in combinations(range(n), 2):
    if (i, j) not in bonded:
        a1.plot(*zip(P[i], P[j]), color=AMBER_LT, lw=0.7, ls=(0, (2.0, 1.8)), zorder=1)
for u, v in edges:
    a1.plot(*zip(P[u], P[v]), color=SLATE, lw=1.9, zorder=2, solid_capstyle='round')
a1.plot(*zip(P[0], P[4]), color=AMBER, lw=1.4, ls=(0, (2.6, 1.8)), zorder=3)
m = (P[0] + P[4]) / 2
a1.text(*m, '%.2f Å\n(3 bonds apart)' % D[0, 4], color=AMBER, fontsize=6.4,
        ha='center', va='center', zorder=7, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.13', fc=PAPER, ec='none', alpha=0.94))
nodes(a1); a1.margins(0.20)
a1.set_title('(a) every pair contributes an entry,\nbonded (solid) or not (dashed)',
             fontsize=7.4, color=SLATE_DK, pad=2)
cmap = LinearSegmentedColormap.from_list('amb', [AMBER_PALE, AMBER_LT, AMBER])
grid(a2, D, D > 0, cmap, 1.0, 4.3, lambda v: f'{v:.2f}', lambda i, j: sp[i][j] == 1)
a2.set_title('(b) $D(G)=[d_{ij}]_{n\\times n}$ in Å;  bold = bonded',
             fontsize=7.4, color=SLATE_DK, pad=3)
fig.savefig('c4b_dist_col.png', dpi=400, facecolor=PAPER, bbox_inches='tight')

# ---------------------------------------------------------------- angle
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.45, 2.52),
                             gridspec_kw=dict(height_ratios=[1.18, 1.0], hspace=0.15))
fig.subplots_adjust(left=0.13, right=0.985, top=0.935, bottom=0.055)
a1.set_aspect('equal'); a1.axis('off')
for u, v in edges:
    a1.plot(*zip(P[u], P[v]), color=SLATE, lw=1.9, zorder=1, solid_capstyle='round')
pairs_by_k = {}
for (i, j), k in sorted(central.items()):
    pairs_by_k.setdefault(k, []).append((i, j))
PAL = [RUST, TEAL]; CCOL = {k: PAL[c % 2] for c, k in enumerate(sorted(pairs_by_k))}
for k, prs in pairs_by_k.items():
    col = CCOL[k]
    for ri, (i, j) in enumerate(prs):
        rad = 0.36 + 0.21 * ri
        v1, v2 = P[i] - P[k], P[j] - P[k]
        a1_, a2_ = (np.degrees(np.arctan2(*v1[::-1])), np.degrees(np.arctan2(*v2[::-1])))
        if (a2_ - a1_) % 360 > 180:
            a1_, a2_ = a2_, a1_
        a1.add_patch(Arc(P[k], 2 * rad, 2 * rad, theta1=a1_, theta2=a2_, color=col,
                         lw=1.2, zorder=3))
        mid = np.radians(a1_ + ((a2_ - a1_) % 360) / 2)
        tp = P[k] + (rad + 0.40) * np.array([np.cos(mid), np.sin(mid)])
        a1.text(*tp, f'{Theta[i, j]:.1f}$^\\circ$', color=col, fontsize=6.2, ha='center',
                va='center', zorder=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.10', fc=PAPER, ec='none', alpha=0.92))
for i in range(n):
    edge = CCOL.get(i, SLATE_DK if sym[i] == 'C' else AMBER_DK)
    face = PAPER if i not in CCOL else ('#F8E7E1' if CCOL[i] == RUST else TEAL_PALE)
    a1.scatter(*P[i], s=210, color=face, edgecolor=edge, linewidth=1.4, zorder=4)
    a1.text(*P[i], lab[i], ha='center', va='center', fontsize=6.8, color=edge, zorder=5)
a1.margins(0.30)
a1.set_title('(a) bond angles at each central atom\n(C$_2$ carries three, C$_3$ one)',
             fontsize=7.4, color=SLATE_DK, pad=2)
cmap2 = LinearSegmentedColormap.from_list('rst', ['#FBE9E3', RUST_LT, RUST])
grid(a2, Theta, Theta > 0, cmap2, 105, 125, lambda v: f'{v:.1f}', lambda i, j: False)
a2.set_title('(b) $\\Theta(G)=[\\theta_{ij}]_{n\\times n}$ in degrees;  grey $0$ = no shared centre',
             fontsize=7.0, color=SLATE_DK, pad=3)
fig.savefig('c4_angle_col.png', dpi=400, facecolor=PAPER, bbox_inches='tight')
print('col figures ok')
