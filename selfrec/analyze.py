"""Self-recognition analysis: does family self-recognition rise with capability?

Per judge: AUROC of the 0-1 score separating OWN-family (positive) vs OTHER-family
(negative) solutions, plus mean own/other and the gap. Then correlate the
recognition metric with each model's LiveBench coding average.
Reads recognition.jsonl. Pure stdlib.
"""
import json, os, bisect, random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = {  # LiveBench coding average
    "claude-sonnet-4.6": 79.98, "claude-sonnet-4.5": 76.07, "claude-haiku-4.5": 72.17,
    "gemini-3.5-flash": 78.18, "gemini-3-flash": 73.90, "gemini-3.1-flash-lite": 68.52,
    "gpt-5.2": 76.45, "gpt-5.4-nano": 69.40, "gpt-5-nano": 52.73,
}


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    s = sorted(neg); w = t = 0
    for p in pos:
        w += bisect.bisect_left(s, p); t += bisect.bisect_right(s, p) - bisect.bisect_left(s, p)
    return (w + 0.5 * t) / (len(pos) * len(neg))


def boot_ci(pos, neg, n=2000, seed=0):
    if not pos or not neg:
        return (float("nan"), float("nan"))
    rng = random.Random(seed); v = []
    for _ in range(n):
        bp = [pos[rng.randrange(len(pos))] for _ in pos]; bn = [neg[rng.randrange(len(neg))] for _ in neg]
        v.append(auroc(bp, bn))
    v.sort(); return v[int(.025 * n)], v[int(.975 * n)]


def pearson(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs); syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else float("nan")


def spearman(xs, ys):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for i, idx in enumerate(o):
            r[idx] = i
        return r
    return pearson(rank(xs), rank(ys))


def main():
    rows = [json.loads(l) for l in open(os.path.join(HERE, "recognition.jsonl"))]
    byj = defaultdict(lambda: {"own": [], "other": []})
    for r in rows:
        if r["score"] is None:
            continue
        byj[r["judge"]]["own" if r["is_own"] else "other"].append(r["score"])

    print(f"{'judge':22}{'cap':>7}{'n own/oth':>11}{'mean_own':>9}{'mean_oth':>9}{'gap':>7}{'AUROC':>7}{'95% CI':>15}")
    res = []
    for j in sorted(byj, key=lambda k: -CAP.get(k, 0)):
        own, oth = byj[j]["own"], byj[j]["other"]
        a = auroc(own, oth); lo, hi = boot_ci(own, oth)
        mo = sum(own) / max(len(own), 1); mt = sum(oth) / max(len(oth), 1)
        res.append((j, CAP[j], a, mo - mt))
        print(f"{j:22}{CAP[j]:7.1f}{f'{len(own)}/{len(oth)}':>11}{mo:9.3f}{mt:9.3f}{mo-mt:+7.3f}{a:7.3f}  [{lo:.3f},{hi:.3f}]")

    caps = [r[1] for r in res]; aurocs = [r[2] for r in res]; gaps = [r[3] for r in res]
    print(f"\ncapability vs AUROC : Pearson {pearson(caps,aurocs):+.3f}  Spearman {spearman(caps,aurocs):+.3f}")
    print(f"capability vs gap   : Pearson {pearson(caps,gaps):+.3f}  Spearman {spearman(caps,gaps):+.3f}")
    # family means
    fam = defaultdict(list)
    for j, c, a, g in res:
        fam[j.split('-')[0] if not j.startswith('gpt') else 'gpt'].append(a)
    print("\nfamily mean AUROC:", {k: round(sum(v) / len(v), 3) for k, v in fam.items()})
    json.dump({"per_judge": [{"judge": j, "cap": c, "auroc": a, "gap": g} for j, c, a, g in res],
               "cap_vs_auroc_spearman": spearman(caps, aurocs)},
              open(os.path.join(HERE, "analysis.json"), "w"), indent=2)
    print("\nwrote analysis.json")


if __name__ == "__main__":
    main()
