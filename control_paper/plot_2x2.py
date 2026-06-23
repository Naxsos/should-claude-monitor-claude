"""Figures for the 2x2 suspicion-scoring experiment.
Fig1: clean-vs-backdoor score distributions per cell (discrimination + saturation).
Fig2: AUROC bar chart with bootstrap CIs, grouped by monitor, same/cross shaded.
Fig3: catch-rate vs audit-budget (clean FPR) curves per cell.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_2x2 import load_cells, auroc, boot_auroc_ci, tp_at_fpr, SAME_FAMILY

LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else "logs/suspicion2x2"
OUT = os.path.join(LOG_DIR, "figs"); os.makedirs(OUT, exist_ok=True)
cells = load_cells(LOG_DIR)
ORDER = [("dsexp","deepseek"),("dsexp","sonnet"),("haiku","deepseek"),("haiku","sonnet")]
def fam(k): return "same-family" if k in SAME_FAMILY else "cross-family"

# Fig 1: distributions
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
for ax, k in zip(axes.flat, ORDER):
    cl, bd = cells[k]["clean"], cells[k]["backdoor"]
    ax.hist(cl, bins=25, range=(0,1), alpha=0.6, label=f"clean (n={len(cl)})", color="#4c72b0", density=True)
    ax.hist(bd, bins=25, range=(0,1), alpha=0.6, label=f"backdoor (n={len(bd)})", color="#c44e52", density=True)
    a = auroc(bd, cl)
    ax.set_title(f"{k[0]} monitor  x  {k[1]} backdoors  [{fam(k)}]\nAUROC={a:.3f}", fontsize=10)
    ax.legend(fontsize=8); ax.set_xlabel("suspicion score (0-1)")
fig.suptitle("Suspicion-score distributions: clean vs successful backdoors", fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_distributions.png", dpi=130); plt.close(fig)

# Fig 2: AUROC with CIs
fig, ax = plt.subplots(figsize=(8, 5))
xs, labels, colors = [], [], []
for i, k in enumerate(ORDER):
    cl, bd = cells[k]["clean"], cells[k]["backdoor"]
    a = auroc(bd, cl); lo, hi = boot_auroc_ci(bd, cl)
    color = "#55a868" if k in SAME_FAMILY else "#8172b3"
    ax.bar(i, a, color=color, alpha=0.85)
    ax.errorbar(i, a, yerr=[[a-lo],[hi-a]], fmt="none", ecolor="black", capsize=4)
    ax.text(i, a+0.012, f"{a:.3f}", ha="center", fontsize=9)
    labels.append(f"{k[0]}\nx {k[1]}")
ax.axhline(0.5, ls="--", color="gray", lw=1, label="chance")
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("AUROC (clean vs backdoor)"); ax.set_ylim(0.45, 0.95)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#55a868", label="same-family"), Patch(color="#8172b3", label="cross-family"),
                   plt.Line2D([],[],ls="--",color="gray",label="chance")], fontsize=9)
ax.set_title("Backdoor-detection AUROC by monitor x generator (95% bootstrap CI)", fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_auroc.png", dpi=130); plt.close(fig)

# Fig 3: catch-rate vs FPR
fig, ax = plt.subplots(figsize=(8, 5))
fprs = [0.01,0.02,0.05,0.10,0.15,0.20,0.30,0.50]
for k in ORDER:
    cl, bd = cells[k]["clean"], cells[k]["backdoor"]
    ys = [tp_at_fpr(bd, cl, f)[0] for f in fprs]
    ls = "-" if k in SAME_FAMILY else "--"
    ax.plot([f*100 for f in fprs], ys, ls, marker="o", label=f"{k[0]} x {k[1]} [{fam(k)[:4]}]")
ax.axvline(2, color="red", ls=":", lw=1, label="2% audit budget")
ax.set_xlabel("audit budget = clean false-positive rate (%)")
ax.set_ylabel("backdoor catch rate (TP)")
ax.set_title("Catch rate vs audit budget — strict budgets catch ~0 (saturated clean tail)", fontweight="bold", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_catch_rate.png", dpi=130); plt.close(fig)

print("wrote figures to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)
