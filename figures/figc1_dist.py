import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from scipy.stats import gaussian_kde
from palette import *
apply_style()
plt.rcParams.update({"font.size": 8.6, "axes.labelsize": 8.8, "xtick.labelsize": 8.0,
                     "ytick.labelsize": 8.0})
df = pd.read_excel('BoilingData.xlsx')
y = df.BP.values.astype(float)
q1, med, q3 = np.percentile(y, [25, 50, 75])
fig, ax = plt.subplots(figsize=(3.45, 2.45))
bins = np.arange(-100, 570, 20)
n, edges, patches = ax.hist(y, bins=bins, color=TEAL_LT, edgecolor=PAPER, linewidth=0.5, zorder=3)
for p, e in zip(patches, edges[:-1]):
    if q1 <= e + 10 <= q3: p.set_facecolor(TEAL)
kde = gaussian_kde(y); xs = np.linspace(y.min()-15, y.max()+15, 500)
ax.plot(xs, kde(xs)*len(y)*(edges[1]-edges[0]), color=SLATE_DK, lw=1.4, zorder=5)
ax.vlines(med, 0, 470, color=AMBER_DK, lw=1.4, zorder=6)
ax.text(med+22, 470, "median\n%.1f" % med + r"$^{\circ}$C", color=AMBER_DK, fontsize=7.8,
        va="top", linespacing=1.25)
ax.set_xlim(-110, 575); ax.set_ylim(0, 620)
ax.set_xlabel("boiling point  (" + r"$^{\circ}$C)")
ax.set_ylabel("compounds")
ax.grid(axis="y", color=SLATE_LT, lw=0.5, alpha=0.7, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.text(0.975, 0.955, "$n$ = 5,432\nrange  $-88.6$ to $548.0$\nSD  85.1" + r"$^{\circ}$C",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.6, color=SLATE, linespacing=1.4)
fig.subplots_adjust(left=0.155, right=0.985, top=0.975, bottom=0.185)
fig.savefig("c1_distribution.png", dpi=400, facecolor=PAPER)
print("c1 ok")
