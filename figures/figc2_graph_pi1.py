import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, networkx as nx, io
import matplotlib.image as mpimg
from matplotlib.patches import Circle, FancyBboxPatch
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from palette import *
from example import load, SHORT
apply_style()
plt.rcParams.update({"font.size": 7.0})

m, pos, bp = load()
syms = [a.GetSymbol() for a in m.GetAtoms()]
ECOL = {"C": SLATE_DK, "N": TEAL, "O": AMBER_DK}
G = nx.Graph()
for a in m.GetAtoms(): G.add_node(a.GetIdx())
for b in m.GetBonds(): G.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
P = {i: (float(pos[i, 0]), float(pos[i, 1])) for i in G.nodes}
T = nx.bfs_tree(G, 0).to_undirected()
chords = [e for e in G.edges if not T.has_edge(*e)]
n, mE, c = G.number_of_nodes(), G.number_of_edges(), 1
basis = sorted(nx.minimum_cycle_basis(G), key=len)
cycles = []
for cy in basis:
    H = G.subgraph(cy); order = nx.cycle_basis(H)[0]; cycles.append(order + [order[0]])
GEN = [TEAL, RUST]

def depiction(mol, w=1000, h=800):
    mm = type(mol)(mol); mm.RemoveAllConformers(); AllChem.Compute2DCoords(mm)
    d = rdMolDraw2D.MolDraw2DCairo(w, h); o = d.drawOptions()
    o.bondLineWidth = 4.0; o.fixedFontSize = 58; o.padding = 0.09
    o.additionalAtomLabelPadding = 0.14; o.useBWAtomPalette()
    o.setAtomPalette({-1: (0.149, 0.196, 0.227), 8: (0.690, 0.416, 0.020), 7: (0.071, 0.475, 0.486)})
    o.setBackgroundColour((1, 1, 1, 1))
    rdMolDraw2D.PrepareAndDrawMolecule(d, mm); d.FinishDrawing()
    return mpimg.imread(io.BytesIO(d.GetDrawingText()))

fig = plt.figure(figsize=(3.50, 4.85))
gs = fig.add_gridspec(3, 2, wspace=0.06, hspace=0.52, left=0.012, right=0.988,
                      top=0.930, bottom=0.052)
CX = [0.265, 0.755]; RY = [0.972, 0.655, 0.338]
def head(r, i, t): fig.text(CX[i], RY[r], t, ha="center", va="center", fontsize=7.6, color=SLATE_DK)
def foot(r, i, t): fig.text(CX[i], RY[r] - 0.281, t, ha="center", va="center", fontsize=6.6, color=SLATE)

def skeleton(ax, style, node_s=40):
    for (u, v) in G.edges:
        col, lw, ls = style(u, v)
        ax.plot([P[u][0], P[v][0]], [P[u][1], P[v][1]], color=col, lw=lw, ls=ls, zorder=2)
    for i in G.nodes:
        ax.scatter(*P[i], s=node_s, color=ECOL[syms[i]], zorder=3, edgecolor=PAPER, linewidth=1.0)
    ax.set_aspect("equal"); ax.margins(0.13); ax.axis("off")

ax = fig.add_subplot(gs[0, 0]); ax.axis("off"); ax.imshow(depiction(m), interpolation="antialiased")
head(0, 0, r"$\bf{(a)}$  chemical structure"); foot(0, 0, f"{SHORT},  $T_b$ = {bp:.1f}" + r"$^{\circ}$C")

ax = fig.add_subplot(gs[0, 1])
skeleton(ax, lambda u, v: (SLATE_MD, 1.2, "-"), 52)
for i in G.nodes:
    ax.text(P[i][0], P[i][1], syms[i], ha="center", va="center", fontsize=4.4,
            color="white", fontweight="bold", zorder=4)
head(0, 1, r"$\bf{(b)}$  molecular graph  $G$"); foot(0, 1, f"$n$ = {n},   $m$ = {mE}")

ax = fig.add_subplot(gs[1, 0], projection="3d")
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=16, c=[ECOL[s] for s in syms],
           depthshade=False, edgecolor=PAPER, linewidth=0.6)
for b in m.GetBonds():
    i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
    ax.plot(*zip(pos[i], pos[j]), color=SLATE_LT, lw=0.9, zorder=1)
ax.set_box_aspect((1, 1, 0.8)); ax.view_init(elev=22, azim=-58)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
    pane.pane.set_facecolor((1, 1, 1, 0)); pane.pane.set_edgecolor(SLATE_LT); pane.line.set_color(SLATE_LT)
ax.grid(color=SLATE_LT, lw=0.35)
head(1, 0, r"$\bf{(c)}$  point cloud  $X$"); foot(1, 0, "heavy-atom coordinates")

ax = fig.add_subplot(gs[1, 1])
skeleton(ax, lambda u, v: (AMBER, 2.3, (0, (3, 2))) if not T.has_edge(u, v) else (SLATE_MD, 1.5, "-"))
for (u, v) in chords:
    ax.scatter([(P[u][0]+P[v][0])/2], [(P[u][1]+P[v][1])/2], s=26, facecolor=PAPER,
               edgecolor=AMBER, linewidth=1.4, zorder=4)
head(1, 1, r"$\bf{(d)}$  spanning tree and chords"); foot(1, 1, f"{n-c} tree edges, {len(chords)} chords")

ax = fig.add_subplot(gs[2, 0])
skeleton(ax, lambda u, v: (SLATE_LT, 1.3, "-"))
for k, cyc in enumerate(cycles):
    xs = [P[i][0] for i in cyc]; ys = [P[i][1] for i in cyc]
    ax.plot(xs, ys, color=GEN[k], lw=2.0, zorder=4 + k, solid_capstyle="round",
            ls="-" if k == 0 else (0, (5, 3)))
    ax.text(np.mean(xs[:-1]), np.mean(ys[:-1]), r"$\gamma_%d$" % (k + 1), color=GEN[k],
            fontsize=8.0, fontweight="bold", ha="center", va="center", zorder=6)
head(2, 0, r"$\bf{(e)}$  free generators"); foot(2, 0, "short cycle-basis representatives")

ax = fig.add_subplot(gs[2, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.add_patch(Circle((0.31, 0.68), 0.165, facecolor="none", edgecolor=GEN[0], lw=2.0, zorder=3))
ax.add_patch(Circle((0.65, 0.68), 0.165, facecolor="none", edgecolor=GEN[1], lw=2.0, zorder=3))
ax.scatter([0.48], [0.68], s=22, color=SLATE_DK, zorder=4)
ax.text(0.31, 0.68, r"$\gamma_1$", color=GEN[0], fontsize=8, ha="center", va="center", fontweight="bold")
ax.text(0.65, 0.68, r"$\gamma_2$", color=GEN[1], fontsize=8, ha="center", va="center", fontweight="bold")
ax.add_patch(FancyBboxPatch((0.02, 0.18), 0.96, 0.29, boxstyle="round,pad=0,rounding_size=0.05",
                            facecolor=TEAL_PALE, edgecolor=TEAL_LT, lw=0.8, zorder=2))
ax.text(0.5, 0.385, r"$G \simeq S^1 \vee S^1$", ha="center", va="center", fontsize=7.6, color=TEAL_DK, zorder=4)
ax.text(0.5, 0.255, r"$\pi_1(G)\cong F_2$", ha="center", va="center", fontsize=7.6, color=TEAL_DK, zorder=4)
head(2, 1, r"$\bf{(f)}$  homotopy type"); foot(2, 1, r"rank $r(G)=m-n+c=%d$" % (mE - n + c))
fig.savefig("c2_graph_pi1.png", dpi=400, facecolor=PAPER, bbox_inches="tight")
print("c2 ok")
