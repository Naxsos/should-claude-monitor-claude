"""Self-recognition vs ALL LiveBench capability dimensions.

The original analyze.py correlated per-judge self-recognition (AUROC of the 0-1
score separating OWN-family from OTHER-family solutions) against a single
capability column: LiveBench *coding* average. This script extends that to every
LiveBench dimension (Global, Reasoning, Coding, Agentic Coding, Mathematics,
Data Analysis, Language, Instruction-Following) to see whether any capability
predicts self-recognition better than coding did.

Recognition metric is computed identically to analyze.py from recognition.jsonl.
Each judge is matched to the exact LiveBench row whose Coding average equals the
coding value used in the original analyze.py CAP dict (so the Coding column here
reproduces the original -0.25 Spearman). Pure stdlib.
"""
import json, os, bisect
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

# 8 LiveBench dimensions per judge. Each row taken from the LiveBench entry whose
# Coding average matches the original analyze.py CAP value for that model:
#   claude-sonnet-4.6     -> Claude 4.6 Sonnet Thinking High Effort
#   claude-sonnet-4.5     -> Claude Sonnet 4.5 (non-thinking)
#   claude-haiku-4.5      -> Claude Haiku 4.5 (non-thinking)
#   gemini-3.5-flash      -> Gemini 3.5 Flash High
#   gemini-3-flash        -> Gemini 3 Flash Preview High
#   gemini-3.1-flash-lite -> Gemini 3.1 Flash Lite Preview High
#   gpt-5.2               -> GPT-5.2 No Thinking (only row with Coding 76.45)
#   gpt-5.4-nano          -> GPT-5.4 Nano Low
#   gpt-5-nano            -> GPT-5 Nano Low
DIMS = ["Global", "Reasoning", "Coding", "Agentic", "Math", "Data", "Language", "IF"]
CAPS = {
    #                       Global  Reason  Coding  Agentic Math    Data    Lang    IF
    "claude-sonnet-4.6":     [75.32, 86.38, 79.98, 56.67, 86.53, 76.06, 77.69, 63.92],
    "claude-sonnet-4.5":     [53.69, 42.29, 76.07, 48.33, 62.62, 47.00, 76.00, 23.52],
    "claude-haiku-4.5":      [45.33, 33.94, 72.17, 33.33, 57.97, 45.13, 57.05, 17.75],
    "gemini-3.5-flash":      [75.02, 82.00, 78.18, 51.67, 88.24, 64.86, 84.58, 75.60],
    "gemini-3-flash":        [72.40, 74.55, 73.90, 40.00, 84.17, 74.77, 84.56, 74.86],
    "gemini-3.1-flash-lite": [61.68, 59.66, 68.52, 33.33, 73.56, 54.90, 73.18, 68.62],
    "gpt-5.2":               [48.91, 42.80, 76.45, 40.00, 58.25, 47.68, 49.97, 27.20],
    "gpt-5.4-nano":          [48.67, 42.13, 69.40, 37.28, 65.16, 43.97, 45.56, 37.18],
    "gpt-5-nano":            [34.34, 27.68, 52.73, 10.00, 48.91, 36.57, 35.40, 29.10],
}


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    s = sorted(neg); w = t = 0
    for p in pos:
        w += bisect.bisect_left(s, p); t += bisect.bisect_right(s, p) - bisect.bisect_left(s, p)
    return (w + 0.5 * t) / (len(pos) * len(neg))


def pearson(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs); syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else float("nan")


def spearman(xs, ys):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
        # average ranks for ties
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[o[k]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def main():
    rows = [json.loads(l) for l in open(os.path.join(HERE, "recognition.jsonl"))]
    byj = defaultdict(lambda: {"own": [], "other": []})
    for r in rows:
        if r["score"] is None:
            continue
        byj[r["judge"]]["own" if r["is_own"] else "other"].append(r["score"])

    # recognition metrics per judge (same as analyze.py)
    judges = sorted(byj, key=lambda k: -CAPS[k][DIMS.index("Coding")])
    auc, gap = {}, {}
    for j in judges:
        own, oth = byj[j]["own"], byj[j]["other"]
        auc[j] = auroc(own, oth)
        gap[j] = sum(own) / max(len(own), 1) - sum(oth) / max(len(oth), 1)

    # per-judge table: capabilities + recognition
    hdr = f"{'judge':22}" + "".join(f"{d:>9}" for d in DIMS) + f"{'AUROC':>8}{'gap':>8}"
    print(hdr)
    for j in judges:
        line = f"{j:22}" + "".join(f"{CAPS[j][i]:9.2f}" for i in range(len(DIMS)))
        print(line + f"{auc[j]:8.3f}{gap[j]:+8.3f}")

    aurocs = [auc[j] for j in judges]
    gaps = [gap[j] for j in judges]

    print(f"\n{'capability':>10}  {'AUROC: Pearson':>16} {'Spearman':>9}   {'gap: Pearson':>14} {'Spearman':>9}")
    summary = []
    for i, d in enumerate(DIMS):
        xs = [CAPS[j][i] for j in judges]
        pa, sa = pearson(xs, aurocs), spearman(xs, aurocs)
        pg, sg = pearson(xs, gaps), spearman(xs, gaps)
        summary.append((d, pa, sa, pg, sg))
        print(f"{d:>10}  {pa:>+16.3f} {sa:>+9.3f}   {pg:>+14.3f} {sg:>+9.3f}")

    # rank dimensions by |Spearman vs AUROC|
    print("\ndimensions ranked by |Spearman (vs AUROC)| — strongest predictor first:")
    for d, pa, sa, pg, sg in sorted(summary, key=lambda r: -abs(r[2])):
        print(f"  {d:>10}  Spearman {sa:+.3f}  (Pearson {pa:+.3f})")

    json.dump({
        "n_judges": len(judges),
        "recognition_metric": "AUROC (own-family vs other-family score), gap = mean_own - mean_other",
        "per_judge": [{"judge": j, "auroc": auc[j], "gap": gap[j],
                       **{DIMS[i]: CAPS[j][i] for i in range(len(DIMS))}} for j in judges],
        "correlations": [{"dim": d, "auroc_pearson": pa, "auroc_spearman": sa,
                          "gap_pearson": pg, "gap_spearman": sg}
                         for d, pa, sa, pg, sg in summary],
    }, open(os.path.join(HERE, "analysis_capabilities.json"), "w"), indent=2)
    print("\nwrote analysis_capabilities.json")


if __name__ == "__main__":
    main()
