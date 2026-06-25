"""3x3 backdoor-detection matrix: monitors {DeepSeek-V3.2-Exp, Haiku 4.5,
GPT-5-nano} x generators {DeepSeek V3.2, Sonnet 4.5, GPT-5.2}.

Reuses the EXACT 0-1 single-score method from suspicion_2x2.py (imported) so
every cell is directly comparable to the existing 2x2 cells. Adds:
  - the GPT-5.2 generator = the three 500-problem generation logs concatenated
    (problems 0-49 + 50-199 + 200-499), with sample ids re-indexed 1..N so the
    same Nth solution is identical across monitors (needed for ensemble joins).
  - GPT-5-nano as the GPT-family monitor (reasoning_effort=minimal, per-sample
    time_limit + fail_on_error=False so a slow/stuck nano sample can't wedge it).

Writes .eval logs to logs/suspicion3x3/ and appends a CELL_SUMMARY (with AUROC)
to logs/suspicion3x3/cell_summaries.json.

Run one cell:   python3 run_3x3.py dsexp:gpt
Smoke (3+3):    SAMPLE_LIMIT=3 python3 run_3x3.py gptnano:gpt
"""
import os, sys, json, time
from pathlib import Path

BASE = Path("/root/ARENA_3.0/capestone/should-claude-monitor-claude/control_paper")
os.chdir(BASE); sys.path.insert(0, str(BASE))

import suspicion_2x2 as b  # reuses build_samples, sus_scorer_01, solver, SONNET/DEEPSEEK logs
import part5_ai_control.solutions as sol
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.solver import generate

LOGDIR = BASE / "logs"
GPT_LOGS = [
    str(LOGDIR / "2026-06-23T11-26-25-00-00_task_6vWr6MBwoQEtons3d65QQK.eval"),  # N=50  (0-49)
    str(LOGDIR / "2026-06-23T12-24-05-00-00_task_hi5peiEvr99qLayYGVcdAY.eval"),  # N=150 (50-199)
    str(LOGDIR / "2026-06-23T12-58-59-00-00_task_9qRvtN6KTwkZzTEvtpZ8rm.eval"),  # N=300 (200-499)
]

MONITORS = dict(b.MONITORS)                       # dsexp, haiku
MONITORS["gptnano"] = "openrouter/openai/gpt-5-nano"

PRICING = {  # $/token (in, out)
    "dsexp": (0.2288e-6, 0.3432e-6),
    "haiku": (1.0e-6, 5.0e-6),
    "gptnano": (0.05e-6, 0.40e-6),
}
# reasoning_effort per monitor (only sent when truthy)
EFFORT = {"dsexp": "none", "haiku": None, "gptnano": "minimal"}

OUT_DIR = BASE / "logs" / "suspicion3x3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_gpt(limit=None):
    """Concatenate the 3 GPT-5.2 logs in fixed order; re-index ids 1..N."""
    samples, nc, nb = [], 0, 0
    for lp in GPT_LOGS:
        for s in sol.eval_log_to_detection_samples(lp):
            lbl = (s.metadata or {}).get("label")
            if limit is not None:
                if lbl == "clean" and nc >= limit:
                    continue
                if lbl == "backdoor" and nb >= limit:
                    continue
            if lbl == "clean":
                nc += 1
            elif lbl == "backdoor":
                nb += 1
            s.metadata["generator"] = "gpt"
            samples.append(s)
    for i, s in enumerate(samples):
        s.id = i + 1
    return samples


def build_generator(gen, limit=None):
    cap = (limit, limit) if limit else (None, None)
    if gen == "gpt":
        return build_gpt(limit=limit)
    if gen == "sonnet":
        return b.build_samples(b.SONNET_LOG, "sonnet", max_clean=cap[0], max_backdoor=cap[1])
    if gen == "deepseek":
        return b.build_samples(b.DEEPSEEK_LOG, "deepseek", max_clean=cap[0], max_backdoor=cap[1])
    raise ValueError(gen)


def auroc(clean, bd):
    if not clean or not bd:
        return None
    wins = sum((x > c) + 0.5 * (x == c) for x in bd for c in clean)
    return wins / (len(bd) * len(clean))


def detect_task(samples):
    return Task(dataset=samples, plan=[generate(), b.suspiciousness_rating_01()],
                scorer=b.sus_scorer_01())


def run_cell(mon, gen, limit=None, max_conn=20):
    model = MONITORS[mon]
    samples = build_generator(gen, limit=limit)
    n_clean = sum(1 for s in samples if s.metadata["label"] == "clean")
    n_bd = sum(1 for s in samples if s.metadata["label"] == "backdoor")
    print(f"=== CELL monitor={mon} generator={gen} | clean={n_clean} backdoor={n_bd} "
          f"(total {len(samples)}) | model={model} ===", flush=True)
    eff = EFFORT.get(mon)
    extra = {"reasoning_effort": eff} if eff else {}
    t0 = time.time()
    log = inspect_eval(
        detect_task(samples), model=model, log_dir=str(OUT_DIR),
        max_connections=max_conn, max_tokens=1500,
        fail_on_error=False, time_limit=180,   # per-sample wall cap (guards slow nano)
        **extra,
    )[0]
    elapsed = time.time() - t0
    scored = [s for s in (log.samples or []) if s.score is not None]
    clean = [float(s.score.value) for s in scored if (s.metadata or {}).get("label") == "clean" and s.score.value != -1.0]
    bd = [float(s.score.value) for s in scored if (s.metadata or {}).get("label") == "backdoor" and s.score.value != -1.0]
    unparsed = sum(1 for s in scored if s.score.value == -1.0)
    usage = (log.stats.model_usage or {}) if log.stats else {}
    tin = sum(u.input_tokens or 0 for u in usage.values())
    tout = sum(u.output_tokens or 0 for u in usage.values())
    pin, pout = PRICING.get(mon, (0.0, 0.0))
    summ = {
        "monitor": mon, "generator": gen, "model": model,
        "n_clean": len(clean), "n_backdoor": len(bd), "unparsed": unparsed,
        "mean_clean": round(sum(clean) / max(len(clean), 1), 4),
        "mean_backdoor": round(sum(bd) / max(len(bd), 1), 4),
        "auroc": round(auroc(clean, bd), 4) if clean and bd else None,
        "tokens_in": tin, "tokens_out": tout,
        "cost_usd": round(tin * pin + tout * pout, 4),
        "elapsed_sec": round(elapsed, 1), "log": str(getattr(log, "location", "")),
    }
    print("CELL_SUMMARY " + json.dumps(summ), flush=True)
    out = OUT_DIR / "cell_summaries.json"
    prev = json.loads(out.read_text()) if out.exists() else []
    prev.append(summ)
    out.write_text(json.dumps(prev, indent=2))
    print(f"appended to {out} ({len(prev)} cells)", flush=True)
    return summ


if __name__ == "__main__":
    cells = sys.argv[1:] or ["gptnano:gpt"]
    limit = int(os.getenv("SAMPLE_LIMIT", "0")) or None
    for arg in cells:
        mon, gen = arg.split(":")
        run_cell(mon, gen, limit=limit)
