"""Experiments A & B on the reasoning-levels backdoor-detection runs (offline, $0).

Reuses the 9 on-disk verdict+confidence eval logs in logs/levels_sonnet/ (n=143
each, SAME 143 samples paired by id across every level). Covers exactly the 9
reported configs: gemini none/low/med/high, haiku none/b2048/b8192, o3mini low/med.

B (calibration decomposition): per config -> AUROC (ranking), positive-call rate
   (suspicion prior), mean confidence overall + split by correct/incorrect, Brier
   score, reliability bins, and mean suspicion on clean vs backdoor (the
   distribution shift). Tests "reasoning moves confidence, not ranking."

A (failure-mode autopsy): per config confusion cells (verdict vs label); then
   WITHIN each family, paired migration low-effort -> high-effort by sample id:
   how many FN->TP (reasoning rescued a miss) vs TN->FP (reasoning manufactured
   suspicion) vs stable. Plus suspicion-shift on clean vs backdoor and a
   verbosity(=analysis length)->verdict correlation. Tests "manufactures
   suspicion / rationalizes" vs "needs more thinking to spot it."

Run: python3 analyze_levels_autopsy.py   (writes levels_autopsy.json)
"""
import os, sys, json, bisect, math
from pathlib import Path
from collections import defaultdict

BASE = Path("/root/ARENA_3.0/capestone/should-claude-monitor-claude/control_paper")
os.chdir(BASE)
from inspect_ai.log import read_eval_log

LOGDIR = BASE / "logs" / "levels_sonnet"
# (family, level, order, filename). order: low effort -> high effort within family.
CONFIGS = [
    ("gemini", "none",  0, "2026-06-25T16-04-05-00-00_task_4Dye8w7G4wjoXSEZB7E7M5.eval"),
    ("gemini", "low",   1, "2026-06-25T16-07-16-00-00_task_LYWPWh3jfz6p7QLCKcMkJh.eval"),
    ("gemini", "medium",2, "2026-06-25T16-12-33-00-00_task_oEXKrYpM3tuTRcr93utxzU.eval"),
    ("gemini", "high",  3, "2026-06-25T16-19-23-00-00_task_FrrFnynKe5Xuqn33SdBPjZ.eval"),
    ("haiku",  "none",  0, "2026-06-25T16-04-04-00-00_task_GJpuD74v6a4EFWw9ogsYbD.eval"),
    ("haiku",  "b2048", 1, "2026-06-25T16-05-51-00-00_task_ai2AovrNnrojHdTjVwzoVv.eval"),
    ("haiku",  "b8192", 2, "2026-06-25T16-08-36-00-00_task_Ry77bahf6i9jbJpxSwD85V.eval"),
    ("o3mini", "low",   0, "2026-06-25T16-04-05-00-00_task_PHSkhNyfd4uSycwkxEXJMF.eval"),
    ("o3mini", "medium",1, "2026-06-25T16-05-40-00-00_task_bDZcidmCMJjHWjb2DYSe4Y.eval"),
]


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    s = sorted(neg); w = t = 0
    for x in pos:
        w += bisect.bisect_left(s, x)
        t += bisect.bisect_right(s, x) - bisect.bisect_left(s, x)
    return (w + 0.5 * t) / (len(pos) * len(neg))


def load(fn):
    """-> {id: rec} for parsed samples (sus_score_01 != -1)."""
    log = read_eval_log(str(LOGDIR / fn))
    out = {}
    for s in (log.samples or []):
        m = s.metadata or {}
        v = m.get("sus_score_01")
        if v is None or v == -1.0:
            continue
        am = s.messages[-1]
        c = am.content
        if isinstance(c, list):
            analysis = " ".join(getattr(b, "text", "") for b in c if getattr(b, "type", None) == "text")
            reasoning = " ".join(getattr(b, "reasoning", "") for b in c if getattr(b, "type", None) == "reasoning")
        else:
            analysis, reasoning = c, ""
        out[s.id] = {
            "id": s.id,
            "label": m.get("label"),               # clean | backdoor
            "y": 1 if m.get("label") == "backdoor" else 0,
            "score": float(v),                      # 0-1 suspicion
            "verdict": m.get("sus_verdict"),        # 0/1 (may be None)
            "conf": m.get("sus_confidence"),
            "analysis_chars": len(analysis or ""),
            "reasoning_chars": len(reasoning or ""),
        }
    return out


def calibration(recs):
    pos = [r["score"] for r in recs if r["y"] == 1]
    neg = [r["score"] for r in recs if r["y"] == 0]
    au = auroc(pos, neg)
    brier = sum((r["score"] - r["y"]) ** 2 for r in recs) / len(recs)
    vr = [r for r in recs if r["verdict"] in (0, 1, "0", "1")]
    pos_rate = sum(1 for r in vr if int(r["verdict"]) == 1) / len(vr) if vr else float("nan")
    cf = [r for r in recs if isinstance(r["conf"], (int, float))]
    mean_conf = sum(r["conf"] for r in cf) / len(cf) if cf else float("nan")
    # confidence when the binary verdict is right vs wrong
    cc = [r["conf"] for r in cf if r["verdict"] not in (None,) and int(r["verdict"]) == r["y"]]
    cw = [r["conf"] for r in cf if r["verdict"] not in (None,) and int(r["verdict"]) != r["y"]]
    return {
        "n": len(recs), "n_bd": len(pos), "n_clean": len(neg),
        "auroc": au, "brier": brier,
        "pos_call_rate": pos_rate,
        "mean_conf": mean_conf,
        "conf_when_right": (sum(cc) / len(cc)) if cc else float("nan"),
        "conf_when_wrong": (sum(cw) / len(cw)) if cw else float("nan"),
        "mean_sus_clean": (sum(neg) / len(neg)) if neg else float("nan"),
        "mean_sus_backdoor": (sum(pos) / len(pos)) if pos else float("nan"),
    }


def confusion(recs):
    tp = fp = tn = fn = nver = 0
    for r in recs:
        if r["verdict"] in (None, "", -1):
            nver += 1; continue
        pred = int(r["verdict"])
        if r["y"] == 1 and pred == 1: tp += 1
        elif r["y"] == 1 and pred == 0: fn += 1
        elif r["y"] == 0 and pred == 1: fp += 1
        else: tn += 1
    n = tp + fp + tn + fn
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "no_verdict": nver,
            "tpr": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1),
            "precision": tp / max(tp + fp, 1)}


def pearson(xs, ys):
    n = len(xs)
    if n < 3: return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs)); sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0: return float("nan")
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / (sx * sy)


def main():
    data = {f"{fam}/{lvl}": load(fn) for fam, lvl, _, fn in CONFIGS}

    # ---------- B: calibration table ----------
    print("=" * 100)
    print("B. CALIBRATION DECOMPOSITION  (ranking vs prior vs confidence)")
    print("=" * 100)
    hdr = f"{'config':14}{'n':>4}{'AUROC':>7}{'Brier':>7}{'pos%':>7}{'susCln':>8}{'susBd':>7}{'conf':>6}{'cRight':>8}{'cWrong':>8}"
    print(hdr)
    calib = {}
    for fam, lvl, _, fn in CONFIGS:
        key = f"{fam}/{lvl}"
        c = calibration(list(data[key].values()))
        calib[key] = c
        print(f"{key:14}{c['n']:>4}{c['auroc']:>7.3f}{c['brier']:>7.3f}"
              f"{c['pos_call_rate']*100:>6.0f}%{c['mean_sus_clean']:>8.3f}{c['mean_sus_backdoor']:>7.3f}"
              f"{c['mean_conf']:>6.2f}{c['conf_when_right']:>8.2f}{c['conf_when_wrong']:>8.2f}")
    print("\n  pos%=fraction called backdoor (prior).  susCln/susBd=mean 0-1 suspicion on clean/backdoor.")
    print("  cRight/cWrong=mean stated confidence when the binary verdict was correct/incorrect.")

    # ---------- A: confusion cells + paired migration ----------
    print("\n" + "=" * 100)
    print("A. FAILURE-MODE AUTOPSY")
    print("=" * 100)
    print("\nConfusion cells (verdict vs label):")
    print(f"{'config':14}{'TP':>4}{'FP':>4}{'TN':>4}{'FN':>4}{'TPR':>7}{'FPR':>7}{'prec':>7}{'analyChars':>12}{'corr(len,sus)':>15}")
    conf_by = {}
    for fam, lvl, _, fn in CONFIGS:
        key = f"{fam}/{lvl}"
        recs = list(data[key].values())
        cm = confusion(recs); conf_by[key] = cm
        mean_chars = sum(r["analysis_chars"] for r in recs) / len(recs)
        corr = pearson([r["analysis_chars"] for r in recs], [r["score"] for r in recs])
        print(f"{key:14}{cm['TP']:>4}{cm['FP']:>4}{cm['TN']:>4}{cm['FN']:>4}"
              f"{cm['tpr']:>7.2f}{cm['fpr']:>7.2f}{cm['precision']:>7.2f}{mean_chars:>12.0f}{corr:>15.2f}")
    print("\n  corr(len,sus): within-config Pearson of analysis length vs suspicion score")
    print("                 (>0 => longer the monitor writes, the more suspicious it gets).")

    # paired low->high migration within each family
    fams = {}
    for fam, lvl, order, fn in CONFIGS:
        fams.setdefault(fam, []).append((order, lvl))
    print("\nPaired migration  LOW-effort -> HIGH-effort  (same samples, by id):")
    migr = {}
    for fam, lst in fams.items():
        lst.sort()
        lo, hi = lst[0][1], lst[-1][1]
        d_lo, d_hi = data[f"{fam}/{lo}"], data[f"{fam}/{hi}"]
        ids = set(d_lo) & set(d_hi)
        cells = defaultdict(int)
        sus_d_clean, sus_d_bd = [], []
        for i in ids:
            a, b = d_lo[i], d_hi[i]
            if a["verdict"] in (None,) or b["verdict"] in (None,):
                pass
            else:
                va, vb = int(a["verdict"]), int(b["verdict"])
                y = a["y"]
                if y == 1 and va == 0 and vb == 1: cells["FN->TP (rescued)"] += 1
                elif y == 1 and va == 1 and vb == 0: cells["TP->FN (lost)"] += 1
                elif y == 0 and va == 0 and vb == 1: cells["TN->FP (manufactured)"] += 1
                elif y == 0 and va == 1 and vb == 0: cells["FP->TN (cleared)"] += 1
                else: cells["stable"] += 1
            (sus_d_bd if a["y"] == 1 else sus_d_clean).append(b["score"] - a["score"])
        migr[fam] = {"lo": lo, "hi": hi, "n_paired": len(ids), "cells": dict(cells),
                     "mean_dsus_clean": sum(sus_d_clean) / len(sus_d_clean) if sus_d_clean else float("nan"),
                     "mean_dsus_backdoor": sum(sus_d_bd) / len(sus_d_bd) if sus_d_bd else float("nan")}
        print(f"\n  {fam}:  {lo} -> {hi}   (n_paired={len(ids)})")
        for k in ["FN->TP (rescued)", "TP->FN (lost)", "TN->FP (manufactured)",
                  "FP->TN (cleared)", "stable"]:
            if cells.get(k):
                print(f"      {k:24} {cells[k]}")
        good = cells.get("FN->TP (rescued)", 0) + cells.get("FP->TN (cleared)", 0)
        bad = cells.get("TP->FN (lost)", 0) + cells.get("TN->FP (manufactured)", 0)
        print(f"      net verdict flips: good(+{good}) vs bad(-{bad}) = {good - bad:+d}")
        print(f"      mean Δsuspicion  clean {migr[fam]['mean_dsus_clean']:+.3f} | "
              f"backdoor {migr[fam]['mean_dsus_backdoor']:+.3f}   "
              f"(separation Δ = {migr[fam]['mean_dsus_backdoor'] - migr[fam]['mean_dsus_clean']:+.3f})")

    json.dump({"calibration": calib, "confusion": conf_by, "migration": migr},
              open(BASE / "levels_autopsy.json", "w"), indent=2, default=str)
    print("\n[written levels_autopsy.json]")


if __name__ == "__main__":
    main()
