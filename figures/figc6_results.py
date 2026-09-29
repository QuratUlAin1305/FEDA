import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, json
from matplotlib.lines import Line2D
from palette import *
apply_style()
plt.rcParams.update({"font.size": 8.6, "axes.labelsize": 8.8, "xtick.labelsize": 8.0,
                     "ytick.labelsize": 9.0})
res = json.load(open('fig_results.json'))
ROWS = [("CONCAT", "FEDA"), ("ESPH", "ESPH"), ("ADM", "DM"), ("AM", "AM"), ("FG", "FG"), ("MACCS", "MACCS")]

fig, ax = plt.subplots(figsize=(3.45, 2.5))
for i, (key, lab) in enumerate(ROWS):
    xg = res[key]["XGBoost"]["r2"]; an = res[key]["ANN"]["r2"]
    lo, hi = min(xg, an), max(xg, an)
    best = res[key]["XGBoost" if xg >= an else "ANN"]
    ax.plot([lo, hi], [i, i], color=SLATE_LT, lw=2.6, zorder=2, solid_capstyle="round")
    ax.scatter([xg], [i], s=46, color=TEAL, zorder=4, edgecolor=PAPER, linewidth=0.9)
    ax.scatter([an], [i], s=46, color=AMBER, zorder=4, edgecolor=PAPER, linewidth=0.9)
    bold = (key == "CONCAT")
    ax.text(hi + 0.012, i + 0.16, f"{hi:.4f}", va="center", ha="left", fontsize=8.4,
            color=SLATE_DK, fontweight="bold" if bold else "normal", zorder=5)
    ax.text(hi + 0.012, i - 0.20, f"{best['rmse']:.1f}" + r"$^{\circ}$C", va="center", ha="left",
            fontsize=7.4, color=SLATE_MD, zorder=5)
ax.axhspan(-0.5, 0.5, color=TEAL_PALE, zorder=0)
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([l for _, l in ROWS])
for t, (k, _) in zip(ax.get_yticklabels(), ROWS):
    if k == "CONCAT": t.set_fontweight("bold"); t.set_color(TEAL_DK)
ax.set_xlim(0.575, 1.045); ax.set_xticks(np.arange(0.6, 1.001, 0.1))
ax.set_ylim(-0.7, 5.75)
ax.axhline(4.5, color=SLATE_MD, lw=0.6, ls=(0,(3,2)), zorder=1)
ax.text(1.04, 4.62, "fingerprint baseline", ha="right", va="bottom", fontsize=7.2, color=SLATE_MD, style="italic")
ax.set_xlabel(r"mean five-fold  $R^2$")
ax.grid(axis="x", color=SLATE_LT, lw=0.5, alpha=0.75, zorder=1); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
handles = [Line2D([], [], marker="o", ls="", color=TEAL, markersize=5.2, label="XGBoost"),
           Line2D([], [], marker="o", ls="", color=AMBER, markersize=5.2, label="ANN")]
ax.legend(handles=handles, frameon=False, fontsize=8.0, ncol=2, loc="upper right",
          bbox_to_anchor=(1.015, 1.03), handletextpad=0.3, columnspacing=1.1)
fig.subplots_adjust(left=0.175, right=0.995, top=0.985, bottom=0.15)
fig.savefig("c6_results.png", dpi=400, facecolor=PAPER)
print("c6 ok")
