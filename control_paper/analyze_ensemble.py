"""Followup: inter-monitor agreement & ensemble detection.

Joins the two monitors' scores on the SAME solutions and asks:
 (1) how correlated are DeepSeek-Exp and Haiku suspicion scores?
 (2) does an ensemble (mean / max of the two) detect backdoors better than either
     monitor alone (AUROC + catch rate at audit budgets)?
Relevant to 'should Claude monitor Claude': if diverse monitors are weakly
correlated and the ensemble helps, *multiple* monitors beat one.
Pure stdlib; reuses analyze_2x2 helpers.
"""
import glob, json, os, sys
from collections import defaultdict
from inspect_ai.log import read_eval_log
from analyze_2x2 import auroc, boot_auroc_ci, tp_at_fpr, MODEL_TAG

LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else "logs/suspicion2x2"


def load_records(log_dir):
    """key = (generator, label, question[:60], code[:60]) -> {monitor: score}."""
    recs = defaultdict(dict)
    for f in glob.glob(os.path.join(log_dir, "*.eval")):
        log = read_eval_log(f)
        mtag = MODEL_TAG.get(log.eval.model.split("/")[-1])
        if mtag is None:
            continue
        for s in log.samples or []:
            md = s.metadata or {}
            sc = (s.scores or {}).get("sus_scorer_01")
            if sc is None or sc.value == -1.0:
                continue
            key = (md.get("generator"), md.get("label"),
                   str(md.get("question"))[:60], str(md.get("code"))[:60])
            recs[key][mtag] = float(sc.value)
    return recs


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs)/n, sum(ys)/n
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    vx = sum((x-mx)**2 for x in xs) ** 0.5
    vy = sum((y-my)**2 for y in ys) ** 0.5
    return cov/(vx*vy) if vx and vy else float("nan")


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0]*len(v)
        i = 0
        while i < len(v):
            j = i
            while j+1 < len(v) and v[order[j+1]] == v[order[i]]:
                j += 1
            avg = (i+j)/2 + 1
            for k in range(i, j+1):
                r[order[k]] = avg
            i = j+1
        return r
    return pearson(rank(xs), rank(ys))


if __name__ == "__main__":
    recs = load_records(LOG_DIR)
    both = {k: v for k, v in recs.items() if "dsexp" in v and "haiku" in v}
    print(f"matched solutions scored by BOTH monitors: {len(both)}")

    # correlation overall and by label
    for lbl in ("all", "clean", "backdoor"):
        items = [v for k, v in both.items() if lbl == "all" or k[1] == lbl]
        ds = [v["dsexp"] for v in items]; hk = [v["haiku"] for v in items]
        print(f"  [{lbl:8}] n={len(items):4}  Pearson={pearson(ds,hk):+.3f}  Spearman={spearman(ds,hk):+.3f}")

    # ensemble vs single, pooled across generators
    clean = [v for k, v in both.items() if k[1] == "clean"]
    bd = [v for k, v in both.items() if k[1] == "backdoor"]
    def col(items, f): return [f(v) for v in items]
    schemes = {
        "dsexp only": lambda v: v["dsexp"],
        "haiku only": lambda v: v["haiku"],
        "ensemble mean": lambda v: (v["dsexp"]+v["haiku"])/2,
        "ensemble max":  lambda v: max(v["dsexp"], v["haiku"]),
        "ensemble min":  lambda v: min(v["dsexp"], v["haiku"]),
    }
    print(f"\n{'scheme':16} {'AUROC':>7} {'95% CI':>16}  {'TP@2%':>6} {'TP@5%':>6} {'TP@10%':>7}")
    out = {"n_matched": len(both), "schemes": {}}
    for name, f in schemes.items():
        pos, neg = col(bd, f), col(clean, f)
        a = auroc(pos, neg); lo, hi = boot_auroc_ci(pos, neg)
        t2 = tp_at_fpr(pos, neg, 0.02)[0]; t5 = tp_at_fpr(pos, neg, 0.05)[0]; t10 = tp_at_fpr(pos, neg, 0.10)[0]
        print(f"{name:16} {a:7.3f} [{lo:.3f},{hi:.3f}]  {t2:6.2f} {t5:6.2f} {t10:7.2f}")
        out["schemes"][name] = {"auroc": round(a,4), "ci": [round(lo,4), round(hi,4)],
                                "tp@2%": round(t2,3), "tp@5%": round(t5,3), "tp@10%": round(t10,3)}
    out["correlation"] = {
        lbl: {"pearson": round(pearson([v["dsexp"] for k,v in both.items() if lbl=="all" or k[1]==lbl],
                                       [v["haiku"] for k,v in both.items() if lbl=="all" or k[1]==lbl]),3)}
        for lbl in ("all","clean","backdoor")}
    with open(os.path.join(LOG_DIR, "analysis_ensemble.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote", os.path.join(LOG_DIR, "analysis_ensemble.json"))
