"""Generate all result figures for zu_experiment_results.md -> figures/*.png.
Reproducible: trajectory numbers computed live from monres_*.jsonl; code-backdoor
AUROC ladder + 3x3 grid use the canonical full-N report values (full .eval logs
gitignored); self-rec + migration loaded from json artifacts.
Run: python3 make_figures.py
"""
import json, glob, os, bisect, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "figures"); os.makedirs(FIG, exist_ok=True)
TRAJ = os.path.join(BASE, "trajectories")
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})
COL = {"gemini": "#E8820C", "haiku": "#7B4FBF", "o3mini": "#D62728",
       "gpt52": "#2CA02C", "gptnano": "#8C8C8C", "gptmini": "#8C564B"}


def auroc(pos, neg):
    if not pos or not neg: return float("nan")
    s = sorted(neg); w = t = 0
    for x in pos:
        w += bisect.bisect_left(s, x); t += bisect.bisect_right(s, x) - bisect.bisect_left(s, x)
    return (w + 0.5 * t) / (len(pos) * len(neg))


def residualize(score, x):
    n = len(x); mx = sum(x) / n; ms = sum(score) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    b = sum((x[i] - mx) * (score[i] - ms) for i in range(n)) / sxx if sxx else 0.0
    a = ms - b * mx
    return [score[i] - (a + b * x[i]) for i in range(n)]


def pear(a, b):
    n = len(a); ma = sum(a) / n; mb = sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a)); sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (sa * sb) if sa and sb else float("nan")


def ece(preds, ys, nbins=10):
    from collections import defaultdict
    b = defaultdict(list)
    for p, y in zip(preds, ys): b[min(int(p * nbins), nbins - 1)].append((p, y))
    e = 0.0; N = len(preds)
    for items in b.values():
        conf = sum(p for p, _ in items) / len(items); acc = sum(y for _, y in items) / len(items)
        e += abs(acc - conf) * len(items) / N
    return e


def load_traj():
    """-> per-config dict with resid AUROC, raw, ece, corr(score,tc), delta."""
    tc = {}
    for l in open(os.path.join(TRAJ, "monitorbench_56x56_actiononly.jsonl")):
        r = json.loads(l); tc[r["uid"]] = max(r["transcript"].count("<tool_call"), r["transcript"].count("[ACTION]"))
    out = {}
    for f in glob.glob(os.path.join(TRAJ, "monres_*.jsonl")):
        rs = [json.loads(l) for l in open(f)]; rs = [r for r in rs if r.get("score") is not None]
        if not rs: continue
        cfg = rs[0]["config"]
        sc = [r["score"] for r in rs]; x = [tc.get(r["uid"], 0) for r in rs]
        y = [1 if r["label"] == "attack" else 0 for r in rs]
        rr = residualize(sc, x)
        raw = auroc([sc[i] for i in range(len(rs)) if y[i]], [sc[i] for i in range(len(rs)) if not y[i]])
        res = auroc([rr[i] for i in range(len(rs)) if y[i]], [rr[i] for i in range(len(rs)) if not y[i]])
        out[cfg] = {"raw": raw, "resid": res, "ece": ece(sc, y), "corr": pear(sc, x),
                    "fam": cfg.split(":")[0], "lvl": cfg.split(":")[1]}
    return out


# ---------------------------------------------------------------- Fig 1: 3x3 grid
def fig1():
    mons = ["DeepSeek-V3.2", "Haiku-4.5", "GPT-5-nano"]
    gens = ["DeepSeek", "Sonnet", "GPT"]
    G = [[0.892, 0.717, 0.724], [0.781, 0.751, 0.733], [0.904, 0.691, 0.807]]  # report grid; * diag
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    im = ax.imshow(G, cmap="RdYlGn", vmin=0.6, vmax=0.95)
    for i in range(3):
        for j in range(3):
            star = "*" if i == j else ""
            ax.text(j, i, f"{G[i][j]:.3f}{star}", ha="center", va="center",
                    fontsize=13, fontweight="bold" if i == j else "normal")
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=2.5))
    ax.set_xticks(range(3), gens); ax.set_yticks(range(3), mons)
    ax.set_xlabel("Generator (backdoor author)"); ax.set_ylabel("Monitor")
    ax.set_title("Exp 1 — 3×3 detection AUROC\n(black box = same family; no diagonal advantage)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="AUROC")
    fig.text(0.5, 0.01, "same−cross interaction +0.022, 95% CI [−0.022, +0.068], p=0.157 (n.s.)",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{FIG}/fig1_3x3_grid.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig 2: self-rec
def fig2():
    d = json.load(open(os.path.join(BASE, "selfrec", "analysis.json")))["per_judge"]
    cap = [j["cap"] for j in d]; acc = [j["auroc"] for j in d]
    FAM = {"claude": ("Anthropic", "#7B4FBF"), "gemini": ("Google", "#E8820C"),
           "gpt": ("OpenAI", "#2CA02C")}
    def fam(name):
        for k, v in FAM.items():
            if name.startswith(k): return v
        return ("other", "#1f77b4")
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.axhline(0.5, color="k", ls="--", lw=1, zorder=1)
    seen = set()
    for j in d:
        label_name, color = fam(j["judge"])
        ax.scatter(j["cap"], j["auroc"], s=80, c=color, zorder=3, edgecolor="white",
                   label=label_name if label_name not in seen else None)
        seen.add(label_name)
        ax.annotate(j["judge"], (j["cap"], j["auroc"]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    r = pear(cap, acc)
    ax.set_xlabel("Judge capability (benchmark score)")
    ax.set_ylabel("Self-family recognition AUROC")
    ax.set_title("Exp 2 — Self-recognition near chance, no capability trend\n"
                 f"Pearson(capability, recognition AUROC) = {r:+.2f}", fontsize=11)
    ax.set_ylim(0.45, 0.70); ax.set_xlim(min(cap) - 3, max(cap) + 4)
    h, l = ax.get_legend_handles_labels()
    h = [plt.Line2D([], [], ls="--", color="k")] + h; l = ["chance (0.50)"] + l
    ax.legend(h, l, loc="upper left", title="family")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_selfrec.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig 3: reasoning vs CODE backdoors
def fig3():
    # canonical full-N AUROC (THINKING_LEVELS_REPORT) + migration from levels_autopsy.json
    ladders = {"gemini": ([0, 1, 2, 3], [0.719, 0.694, 0.691, 0.729]),
               "haiku":  ([0, 1, 2],    [0.721, 0.727, 0.729]),
               "o3mini": ([1, 2],       [0.769, 0.721])}
    labels = ["none", "low/2k", "med/8k", "high"]
    mig = json.load(open(os.path.join(BASE, "control_paper", "levels_autopsy.json")))["migration"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for fam, (xs, ys) in ladders.items():
        a1.plot(xs, ys, "-o", color=COL[fam], label=fam, lw=2, ms=7)
    a1.set_xticks(range(4), labels); a1.set_ylim(0.5, 0.85)
    a1.axhline(0.5, color="gray", ls=":", lw=1)
    a1.set_xlabel("Reasoning effort  (→ more thinking)"); a1.set_ylabel("AUROC (clean vs backdoor)")
    a1.set_title("Detection is FLAT across reasoning effort\n(o3-mini med < low: thinking hurts)")
    a1.legend()
    # migration: manufactured (TN->FP) vs rescued (FN->TP)
    fams = list(mig.keys())
    man = [mig[f]["cells"].get("TN->FP (manufactured)", 0) for f in fams]
    res = [mig[f]["cells"].get("FN->TP (rescued)", 0) for f in fams]
    xpos = range(len(fams)); w = 0.38
    a2.bar([x - w / 2 for x in xpos], man, w, color="#D62728", label="TN→FP  manufactured suspicion")
    a2.bar([x + w / 2 for x in xpos], res, w, color="#2CA02C", label="FN→TP  rescued catch")
    a2.set_xticks(list(xpos), [f"{f}\n({mig[f]['lo']}→{mig[f]['hi']})" for f in fams])
    a2.set_ylabel("# verdict flips (paired, low→high effort)")
    a2.set_title("More reasoning manufactures false alarms\nfar more than it rescues real backdoors")
    a2.legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_reasoning_code.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig 4: reasoning vs TRAJECTORIES
def fig4(traj):
    order = {"none": 0, "minimal": 0, "low": 1, "b2048": 1, "medium": 2, "b8192": 2,
             "high": 3, "b16384": 3, "xhigh": 4, "b32768": 4}
    # show only monitors that clear the length-only baseline; the 3 small OpenAI
    # models (o3-mini, gpt-5-nano, gpt-5-mini) sit at/below it and are omitted.
    KEEP = {"haiku", "gemini", "gpt52"}
    fams = {}
    for cfg, d in traj.items():
        if d["fam"] not in KEEP:
            continue
        fams.setdefault(d["fam"], []).append((order.get(d["lvl"], 9), d["lvl"], d["resid"]))
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for fam, pts in fams.items():
        pts.sort(); xs = [p[0] for p in pts]; ys = [p[2] for p in pts]
        ax.plot(xs, ys, "-o", color=COL.get(fam, "k"), label=fam, lw=2, ms=7)
    ax.axhline(0.62, color="brown", ls="--", lw=1.2, label="length-only baseline (0.62)")
    ax.set_xticks(range(5), ["none/\nminimal", "low/\n2k", "med/\n8k", "high/\n16k", "xhigh/\n32k"])
    ax.set_xlabel("Reasoning effort / thinking budget  (→ more)")
    ax.set_ylabel("Residualized AUROC  (signal beyond #tool_calls)")
    ax.set_title("Exp 4 — On trajectories, reasoning HELPS capable monitors\n"
                 "Haiku peaks at 16k  (3 small OpenAI models below baseline omitted)")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_reasoning_traj.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig 5: capability <-> calibration
def fig5(traj):
    # code-backdoor points (from calib_vs_capability.py): (AUROC, ECE) — TWO separate
    # datasets; correlation is only meaningful WITHIN each set (different base difficulty).
    s3x3 = {"GPT-5-nano": (0.743, 0.160), "DeepSeek": (0.716, 0.377), "Haiku": (0.693, 0.327)}
    slvl = {"Haiku": (0.623, 0.301), "Gemini": (0.617, 0.453), "o3-mini": (0.571, 0.361)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    r3 = pear([v[0] for v in s3x3.values()], [v[1] for v in s3x3.values()])
    rl = pear([v[0] for v in slvl.values()], [v[1] for v in slvl.values()])
    a1.scatter([v[0] for v in s3x3.values()], [v[1] for v in s3x3.values()], s=80, c="#D62728",
               marker="o", zorder=3, edgecolor="white", label=f"3×3 set  (r={r3:+.2f}, nano-driven)")
    a1.scatter([v[0] for v in slvl.values()], [v[1] for v in slvl.values()], s=80, c="#FF9896",
               marker="s", zorder=3, edgecolor="white", label=f"levels set  (r={rl:+.2f})")
    for k, v in s3x3.items(): a1.annotate(k, v, fontsize=7, xytext=(4, 3), textcoords="offset points")
    for k, v in slvl.items(): a1.annotate(k, v, fontsize=7, xytext=(4, -10), textcoords="offset points")
    a1.set_title("CODE backdoors: NO consistent capability↔calibration link")
    a1.set_xlabel("AUROC (capability)"); a1.set_ylabel("ECE  (lower = better calibrated)")
    a1.legend(loc="lower left", fontsize=8)
    # trajectory points
    tx = [d["resid"] for d in traj.values() if not math.isnan(d["resid"])]
    ty = [d["ece"] for d in traj.values() if not math.isnan(d["resid"])]
    a2.scatter(tx, ty, s=70, c="#7B4FBF", zorder=3, edgecolor="white")
    a2.set_title(f"TRAJECTORIES: calibration TIGHTLY tracks capability\nPearson = {pear(tx, ty):+.2f}")
    a2.set_xlabel("Residualized AUROC (capability)")
    for a in (a1, a2): a.axhline(0.05, color="green", ls=":", lw=1)
    a2.text(0.40, 0.065, "well-calibrated (<0.05)", color="green", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig5_calib_coupling.png"); plt.close(fig)


# ---------------------------------------------------------------- Fig 6: residualization mechanism
def fig6(traj):
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for cfg, d in traj.items():
        ax.scatter(d["corr"], d["resid"] - d["raw"], s=55, color=COL.get(d["fam"], "k"),
                   edgecolor="white", zorder=3)
    # legend by family
    for fam in sorted(set(d["fam"] for d in traj.values())):
        ax.scatter([], [], color=COL.get(fam, "k"), label=fam, s=55)
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8, ls=":")
    cs = [d["corr"] for d in traj.values()]; ds = [d["resid"] - d["raw"] for d in traj.values()]
    ax.set_title(f"Why residualization hits some monitors harder\n"
                 f"Δ(resid−raw) tracks how much the score leaned on #tool_calls "
                 f"(r={pear(cs, ds):+.2f})")
    ax.set_xlabel("corr(monitor score, #tool_calls)  ← scores short=attack traces high")
    ax.set_ylabel("AUROC change from residualizing (resid − raw)")
    ax.legend(fontsize=9)
    fig.text(0.5, 0.005, "left (negative corr) = leaned on length → big drop;  "
             "right (≈0/positive) = no drop / gain", ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{FIG}/fig6_residualization.png"); plt.close(fig)


if __name__ == "__main__":
    traj = load_traj()
    fig1(); fig2(); fig3(); fig4(traj); fig5(traj); fig6(traj)
    print("wrote:", sorted(os.listdir(FIG)))
