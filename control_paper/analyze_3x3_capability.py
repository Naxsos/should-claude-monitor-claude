"""Does a monitor uniformly do better/worse against the (more capable) same-family
generator? And does the 3-monitor ensemble uniformly help? Bootstrap CIs included.

Within-family capability relationship (monitor T vs same-family generator U):
  DeepSeek : T=V3.2-Exp(reasoning)  vs  U=V3.2(base)   -> monitor MORE capable
  Claude   : T=Haiku 4.5            vs  U=Sonnet 4.5    -> monitor LESS capable
  GPT      : T=GPT-5-nano           vs  U=GPT-5.2       -> monitor LESS capable

(a) per monitor: AUROC(own-family gen) - mean AUROC(other-family gens), CI by
    independent bootstrap of each cell.
(b) per generator: ensemble(mean of 3 monitors) AUROC - best single monitor AUROC,
    paired bootstrap over shared samples.
Reads logs/suspicion3x3/*.eval via analyze_3x3.load.
"""
import random, json, os
from analyze_3x3 import load, auroc, MONS, GENS, SAME, LOG_DIR

OWN = {"dsexp": "deepseek", "haiku": "sonnet", "gptnano": "gpt"}
CAP = {  # is the monitor more/less capable than its same-family generator?
    "dsexp": "monitor MORE capable (Exp reasoning vs base V3.2)",
    "haiku": "monitor LESS capable (Haiku 4.5 vs Sonnet 4.5)",
    "gptnano": "monitor LESS capable (GPT-5-nano vs GPT-5.2)",
}
B = 3000


def split(cell):
    cl = [v for (l, v) in cell.values() if l == "clean"]
    bd = [v for (l, v) in cell.values() if l == "backdoor"]
    return cl, bd


def resample_auroc(cl, bd, rng):
    bcl = [cl[rng.randrange(len(cl))] for _ in cl]
    bbd = [bd[rng.randrange(len(bd))] for _ in bd]
    return auroc(bbd, bcl)


def main():
    cells = load()

    print("=" * 78)
    print("(a) WITHIN-MONITOR: own-family AUROC minus mean(other-family AUROC)")
    print("=" * 78)
    a_rows = {}
    for m in MONS:
        if (m, OWN[m]) not in cells:
            continue
        own_cl, own_bd = split(cells[(m, OWN[m])])
        others = [g for g in GENS if g != OWN[m] and (m, g) in cells]
        oth = {g: split(cells[(m, g)]) for g in others}
        own_a = auroc(own_bd, own_cl)
        oth_a = {g: auroc(oth[g][1], oth[g][0]) for g in others}
        rng = random.Random(0)
        deltas = []
        for _ in range(B):
            oa = resample_auroc(own_cl, own_bd, rng)
            os_ = [resample_auroc(oth[g][0], oth[g][1], rng) for g in others]
            deltas.append(oa - sum(os_) / len(os_))
        deltas.sort()
        lo, hi = deltas[int(0.025 * B)], deltas[int(0.975 * B)]
        base = own_a - sum(oth_a.values()) / len(oth_a)
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s. (CI spans 0)"
        a_rows[m] = {"own_gen": OWN[m], "own_auroc": own_a, "others": oth_a,
                     "delta": base, "ci": [lo, hi], "sig": sig, "capability": CAP[m]}
        print(f"\n  {m}  [{CAP[m]}]")
        print(f"    own({OWN[m]}) AUROC = {own_a:.3f};  others: " +
              ", ".join(f"{g}={oth_a[g]:.3f}" for g in others))
        print(f"    delta (own - mean others) = {base:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  -> {sig}")

    signs = {m: ("+" if a_rows[m]["delta"] > 0 else "-") for m in a_rows}
    uniform = len(set(signs.values())) == 1
    print("\n  --> direction by monitor:", "  ".join(f"{m}:{s}" for m, s in signs.items()))
    print(f"  --> UNIFORM across monitors? {'YES' if uniform else 'NO'}  "
          f"(signs {'all agree' if uniform else 'DISAGREE'})")
    # capability angle: does sign track monitor-vs-generator capability?
    print("  --> capability pattern: dsexp(monitor stronger)=%s,  haiku/gptnano(monitor weaker)=%s/%s"
          % (signs.get("dsexp"), signs.get("haiku"), signs.get("gptnano")))

    print("\n" + "=" * 78)
    print("(b) ENSEMBLE (mean of 3 monitors) minus BEST single monitor, per generator")
    print("=" * 78)
    b_rows = {}
    for g in GENS:
        if not all((m, g) in cells for m in MONS):
            continue
        ids = set.intersection(*[set(cells[(m, g)]) for m in MONS])
        rows = []
        for sid in ids:
            lbl = cells[(MONS[0], g)][sid][0]
            sc = {m: cells[(m, g)][sid][1] for m in MONS}
            sc["_ens"] = sum(sc[m] for m in MONS) / len(MONS)
            sc["_lbl"] = lbl
            rows.append(sc)
        pos = [r for r in rows if r["_lbl"] == "backdoor"]
        neg = [r for r in rows if r["_lbl"] == "clean"]
        singles = {m: auroc([r[m] for r in pos], [r[m] for r in neg]) for m in MONS}
        best = max(singles, key=lambda k: singles[k])
        ens_a = auroc([r["_ens"] for r in pos], [r["_ens"] for r in neg])
        rng = random.Random(0)
        deltas = []
        for _ in range(B):
            bp = [pos[rng.randrange(len(pos))] for _ in pos]
            bn = [neg[rng.randrange(len(neg))] for _ in neg]
            ea = auroc([r["_ens"] for r in bp], [r["_ens"] for r in bn])
            ba = auroc([r[best] for r in bp], [r[best] for r in bn])
            deltas.append(ea - ba)
        deltas.sort()
        lo, hi = deltas[int(0.025 * B)], deltas[int(0.975 * B)]
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "n.s. (CI spans 0)"
        b_rows[g] = {"singles": singles, "best": best, "ensemble": ens_a,
                     "delta": ens_a - singles[best], "ci": [lo, hi], "sig": sig}
        print(f"\n  gen={g}:  singles " + ", ".join(f"{m}={singles[m]:.3f}" for m in MONS))
        print(f"    ensemble={ens_a:.3f}  vs best single ({best}={singles[best]:.3f})  "
              f"delta={ens_a-singles[best]:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  -> {sig}")

    bsigns = {g: ("+" if b_rows[g]["delta"] > 0 else "-") for g in b_rows}
    buniform = len(set(bsigns.values())) == 1
    print("\n  --> ensemble direction by generator:", "  ".join(f"{g}:{s}" for g, s in bsigns.items()))
    print(f"  --> ensemble UNIFORM? {'YES' if buniform else 'NO'}")

    with open(os.path.join(LOG_DIR, "analysis_3x3_capability.json"), "w") as f:
        json.dump({"within_monitor": a_rows, "ensemble": b_rows,
                   "within_uniform": uniform, "ensemble_uniform": buniform}, f, indent=2)
    print(f"\nwrote {os.path.join(LOG_DIR, 'analysis_3x3_capability.json')}")


if __name__ == "__main__":
    main()
