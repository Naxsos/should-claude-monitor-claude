"""Significance test for the 3x3 same-family interaction.

Reads the per-cell .eval logs in a 3x3 log dir, computes per-cell AUROC + bootstrap
CI, and bootstraps the SAME-minus-CROSS interaction (resampling clean/backdoor
within each cell) to get a CI / sign test on the interaction.

Usage: python3 analyze_3x3_significance.py <log_dir>
"""
import sys, glob, os, bisect, random
from inspect_ai.log import read_eval_log

LOG = sys.argv[1] if len(sys.argv) > 1 else "logs/suspicion3x3_200"
MT = {"deepseek-v3.2-exp": "dsexp", "claude-haiku-4.5": "haiku", "gpt-5-nano": "gptnano"}
MON = ["dsexp", "haiku", "gptnano"]; GEN = ["deepseek", "sonnet", "gpt"]
SAME = {("dsexp", "deepseek"), ("haiku", "sonnet"), ("gptnano", "gpt")}
B = 5000


def auroc(p, n):
    if not p or not n:
        return float("nan")
    s = sorted(n); w = t = 0
    for x in p:
        w += bisect.bisect_left(s, x); t += bisect.bisect_right(s, x) - bisect.bisect_left(s, x)
    return (w + 0.5 * t) / (len(p) * len(n))


def load():
    cells = {}
    for f in glob.glob(os.path.join(LOG, "*.eval")):
        log = read_eval_log(f)
        m = MT.get(log.eval.model.split("/")[-1])
        if not m:
            continue
        for s in log.samples or []:
            sc = (s.scores or {}).get("sus_scorer_01")
            if not sc or sc.value == -1.0:
                continue
            g = (s.metadata or {}).get("generator"); lbl = (s.metadata or {}).get("label")
            if g and lbl in ("clean", "backdoor"):
                cells.setdefault((m, g), {"clean": [], "backdoor": []})[lbl].append(float(sc.value))
    return cells


def main():
    cells = load()
    if len(cells) < 9:
        print(f"WARNING: only {len(cells)}/9 cells present")
    rng = random.Random(0)
    # per-cell AUROC + CI
    print(f"{'cell':22}{'n(C/B)':>10}{'AUROC':>8}{'95% CI':>16}")
    aurocs = {}
    for m in MON:
        for g in GEN:
            c = cells.get((m, g))
            if not c:
                continue
            cl, bd = c["clean"], c["backdoor"]
            a = auroc(bd, cl); aurocs[(m, g)] = a
            boot = sorted(auroc([bd[rng.randrange(len(bd))] for _ in bd],
                                [cl[rng.randrange(len(cl))] for _ in cl]) for _ in range(2000))
            lo, hi = boot[50], boot[1949]
            tag = "*" if (m, g) in SAME else " "
            print(f"  {m+' x '+g:20}{f'{len(cl)}/{len(bd)}':>10}{a:>7.3f}{tag}  [{lo:.3f},{hi:.3f}]")
    same = [aurocs[k] for k in aurocs if k in SAME]; cross = [aurocs[k] for k in aurocs if k not in SAME]
    inter = sum(same) / len(same) - sum(cross) / len(cross)
    # bootstrap the interaction: resample within each cell, recompute
    boot = []
    for _ in range(B):
        a2 = {}
        for (m, g), c in cells.items():
            cl, bd = c["clean"], c["backdoor"]
            a2[(m, g)] = auroc([bd[rng.randrange(len(bd))] for _ in bd],
                               [cl[rng.randrange(len(cl))] for _ in cl])
        s = [a2[k] for k in a2 if k in SAME]; x = [a2[k] for k in a2 if k not in SAME]
        boot.append(sum(s) / len(s) - sum(x) / len(x))
    boot.sort()
    lo, hi = boot[int(.025 * B)], boot[int(.975 * B)]
    p_gt0 = sum(1 for v in boot if v <= 0) / B  # one-sided: P(interaction<=0)
    print(f"\n  SAME mean={sum(same)/len(same):.3f}  CROSS mean={sum(cross)/len(cross):.3f}")
    print(f"  INTERACTION (same - cross) = {inter:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  one-sided p(interaction<=0) = {p_gt0:.3f}")
    print(f"  SIGNIFICANT at 95%: {'YES' if (lo>0 or hi<0) else 'NO (CI spans 0)'}")


if __name__ == "__main__":
    main()
