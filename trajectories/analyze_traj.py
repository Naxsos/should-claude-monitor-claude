"""Analyze trajectory attack-detection, controlling for the #tool_calls confound.

For each monitor config: raw AUROC (attack vs honest) + the de-confounded reads:
  - residualized AUROC  : AUROC of monitor_score after regressing out #tool_calls
  - incremental validity : logistic LR test — does monitor_score add predictive power
                           beyond {n_tool_calls}? (delta AUROC + p-value)
  - paired sign test     : per task_id pair, does the monitor score the attack member
                           higher than its honest partner? (binomial p)
Reads monres_*.jsonl. Pure stdlib + (optional) sklearn for the LR test.
"""
import json, glob, os, bisect, math
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    s = sorted(neg); w = t = 0
    for x in pos:
        w += bisect.bisect_left(s, x); t += bisect.bisect_right(s, x) - bisect.bisect_left(s, x)
    return (w + 0.5 * t) / (len(pos) * len(neg))


def residualize(score, x):
    """Return score - OLS_fit(score ~ x); removes the linear x component."""
    n = len(x); mx = sum(x) / n; ms = sum(score) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    b = sum((x[i] - mx) * (score[i] - ms) for i in range(n)) / sxx if sxx else 0.0
    a = ms - b * mx
    return [score[i] - (a + b * x[i]) for i in range(n)]


def binom_p(k, n, p=0.5):
    # two-sided exact-ish via normal approx (n small -> fine as a guide)
    if n == 0:
        return float("nan")
    z = (k - n * p) / math.sqrt(n * p * (1 - p))
    from math import erf
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / math.sqrt(2))))


def main():
    # real #tool_calls per uid (the dataset jsonl omitted it -> count from transcript)
    tcmap = {}
    for l in open(os.path.join(HERE, "monitorbench_56x56_actiononly.jsonl")):
        r = json.loads(l)
        tcmap[r["uid"]] = r["transcript"].count("<tool_call")
    if max(tcmap.values()) == 0:  # action-only uses [ACTION] markers
        for l in open(os.path.join(HERE, "monitorbench_56x56_actiononly.jsonl")):
            r = json.loads(l)
            tcmap[r["uid"]] = max(r["transcript"].count("<tool_call"), r["transcript"].count("[ACTION]"))
    by_cfg = defaultdict(list)
    for f in glob.glob(os.path.join(HERE, "monres_*.jsonl")):
        for l in open(f):
            d = json.loads(l)
            if d["score"] is not None:
                d["n_tool_calls"] = tcmap.get(d["uid"], 0)  # backfill real count
                by_cfg[d["config"]].append(d)
    order = {"none": 0, "low": 1, "minimal": 1, "medium": 2, "high": 3, "b2048": 1, "b8192": 2}
    print(f"{'config':14}{'n':>5}{'rawAUROC':>10}{'residAUROC':>12}{'incrΔAUROC':>12}{'incr p':>9}{'paired p':>10}")
    rows = {}
    for cfg in sorted(by_cfg, key=lambda c: (c.split(":")[0], order.get(c.split(":")[1], 9))):
        rs = by_cfg[cfg]
        pos = [r["score"] for r in rs if r["label"] == "attack"]
        neg = [r["score"] for r in rs if r["label"] == "honest"]
        raw = auroc(pos, neg)
        # residualize vs n_tool_calls
        sc = [r["score"] for r in rs]; tc = [r["n_tool_calls"] or 0 for r in rs]
        y = [1 if r["label"] == "attack" else 0 for r in rs]
        res = residualize(sc, tc)
        rpos = [res[i] for i in range(len(rs)) if y[i] == 1]; rneg = [res[i] for i in range(len(rs)) if y[i] == 0]
        res_auroc = auroc(rpos, rneg)
        # incremental validity: AUROC(tc only) vs AUROC(tc+score) via logistic, plus LR p
        incr = pval = float("nan")
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            from scipy import stats
            X1 = np.array(tc).reshape(-1, 1); X2 = np.column_stack([tc, sc]); Y = np.array(y)
            ll = []
            for X in (X1, X2):
                m = LogisticRegression(max_iter=1000).fit(X, Y)
                p = m.predict_proba(X)[:, 1].clip(1e-9, 1 - 1e-9)
                ll.append(sum(Y * np.log(p) + (1 - Y) * np.log(1 - p)))
                last_p = p
            incr = roc_auc_score(Y, last_p) - roc_auc_score(Y, LogisticRegression(max_iter=1000).fit(X1, Y).predict_proba(X1)[:, 1])
            lr = 2 * (ll[1] - ll[0]); pval = 1 - stats.chi2.cdf(lr, df=1)
        except Exception:
            pass
        # paired sign test by task_id
        bytask = defaultdict(dict)
        for r in rs:
            bytask[r["task_id"]][r["label"]] = r["score"]
        wins = pairs = 0
        for t, d in bytask.items():
            if "attack" in d and "honest" in d:
                pairs += 1; wins += 1 if d["attack"] > d["honest"] else (0.5 if d["attack"] == d["honest"] else 0)
        pp = binom_p(round(wins), pairs)
        rows[cfg] = (raw, res_auroc, incr, pval, wins, pairs, pp)
        print(f"{cfg:14}{len(rs):>5}{raw:>10.3f}{res_auroc:>12.3f}{incr:>12.3f}{pval:>9.3f}{pp:>10.3f}")
    print("\n  rawAUROC: vs honest (confounded by #tool_calls).  residAUROC: after removing #tool_calls.")
    print("  incrΔAUROC/p: does monitor add signal beyond #tool_calls (LR test).  paired p: attack>honest within task pair.")
    json.dump({c: {"raw": r[0], "resid": r[1], "incr": r[2], "incr_p": r[3], "paired_wins": r[4], "pairs": r[5], "paired_p": r[6]} for c, r in rows.items()},
              open(os.path.join(HERE, "traj_analysis.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
