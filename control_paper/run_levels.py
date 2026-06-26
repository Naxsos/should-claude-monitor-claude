"""0-1 suspicion scoring of the Sonnet 500-run (715 detection samples: 500 clean
+ 215 successful backdoors) by 3 monitors across their reasoning/thinking levels:
  deepseek = DeepSeek-V3.2-Exp     (expected 2 levels: non-thinking / thinking)
  o3mini   = OpenAI o3-mini        (expected 3 levels: low / medium / high)
  gemini   = Gemini 2.5 Flash      (expected 4 levels)

Each (model, effort) is ONE eval log in logs/levels_sonnet/. Reuses the exact 0-1
method from suspicion_2x2 (SUS_PROMPT_01 / sus_scorer_01). Bigger token budgets
than the 2x2 because reasoning monitors must spend tokens thinking AND still emit
the <SCORE> tag (a too-small budget -> empty answer -> unparsed).

Captures reasoning_tokens per run so we can VERIFY which efforts actually engage
thinking and whether gradations collapse (-> the true level count).

Run: SAMPLE_LIMIT=3 python3 run_levels.py deepseek none,low,medium,high
"""
import os, sys, json, time
from pathlib import Path

BASE = Path("/root/ARENA_3.0/capestone/should-claude-monitor-claude/control_paper")
os.chdir(BASE); sys.path.insert(0, str(BASE))

import suspicion_2x2 as b
sys.path.insert(0, str(BASE.parent / "thursday_experiments_zu"))
from prompts import SUS_PROMPT_01, MONITOR_IDENTITY_SYS, developer_from_slug, model_name_from_slug  # reasoning-levels reuses the 0-1 rating prompt
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.solver import solver, generate, system_message, Solver, TaskState, Generate
from inspect_ai.model import get_model, GenerateConfig, ChatMessageUser

MODELS = {
    "deepseek": ("openrouter/deepseek/deepseek-v4-flash", 0.09e-6, 0.18e-6),
    "o3mini":   ("openrouter/openai/o3-mini",            1.10e-6, 4.40e-6),
    "gemini":   ("openrouter/google/gemini-2.5-flash",   0.30e-6, 2.50e-6),
    "gpt5":     ("openrouter/openai/gpt-5",              1.25e-6, 10.0e-6),
    "haiku":    ("openrouter/anthropic/claude-haiku-4.5", 1.0e-6, 5.0e-6),
    "gpt52":    ("openrouter/openai/gpt-5.2",            1.75e-6, 14.0e-6),
}
OUT = BASE / "logs" / "levels_sonnet"
OUT.mkdir(parents=True, exist_ok=True)
INV_MAXTOK = int(os.getenv("INV_MAXTOK", "18000"))   # investigation turn (room to think; > any thinking budget)
SCORE_MAXTOK = int(os.getenv("SCORE_MAXTOK", "4000"))  # score turn (non-thinking; room to emit tag)


@solver
def score01() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        cfg = GenerateConfig(temperature=0.0, max_tokens=SCORE_MAXTOK)
        msgs = list(state.messages) + [ChatMessageUser(content=SUS_PROMPT_01)]
        out = await model.generate(input=msgs, config=cfg)
        state.output = out
        verdict, conf = b.parse_verdict_conf(out.completion)
        v = b.parse_score_01(out.completion)
        state.metadata["sus_score_01"] = v if v is not None else -1.0
        state.metadata["sus_verdict"] = verdict
        state.metadata["sus_confidence"] = conf
        return state
    return solve


def task(samples, identity=None):
    plan = ([system_message(identity)] if identity else []) + [generate(), score01()]
    return Task(dataset=samples, plan=plan, scorer=b.sus_scorer_01())


def run(model_tag, effort, limit):
    model, pin, pout = MODELS[model_tag]
    identity = MONITOR_IDENTITY_SYS.format(model=model_name_from_slug(model),
                                           developer=developer_from_slug(model))
    samples = b.build_samples(b.SONNET_LOG, "sonnet")
    if limit:
        samples = samples[:limit]
    print(f"=== {model_tag} effort={effort!r} | n={len(samples)} | model={model} ===", flush=True)
    kw = {}
    if model_tag == "haiku":
        # Claude: thinking controlled by a token budget. effort "none" = off;
        # "b<N>" = extended thinking with budget N (requires temperature=1).
        if effort.startswith("b"):
            kw["reasoning_tokens"] = int(effort[1:])
            kw["temperature"] = 1.0
        # effort == "none" -> no reasoning kwargs (thinking off, temp default 0)
    elif effort and effort not in ("default", ""):
        kw["reasoning_effort"] = effort
    t0 = time.time()
    try:
        log = inspect_eval(task(samples, identity=identity), model=model, log_dir=str(OUT),
                           max_connections=20, max_tokens=INV_MAXTOK,
                           fail_on_error=False, time_limit=400, **kw)[0]
    except Exception as e:
        print("LEVELSUM " + json.dumps({"model_tag": model_tag, "effort": effort,
              "ERROR": str(e)[:300]}), flush=True)
        return None
    el = time.time() - t0
    scored = [s for s in (log.samples or []) if s.score is not None]
    vals = [float(s.score.value) for s in scored if s.score.value != -1.0]
    unparsed = sum(1 for s in scored if s.score.value == -1.0)
    usage = (log.stats.model_usage or {}) if log.stats else {}
    tin = sum(u.input_tokens or 0 for u in usage.values())
    tout = sum(u.output_tokens or 0 for u in usage.values())
    treason = sum((getattr(u, "reasoning_tokens", 0) or 0) for u in usage.values())
    summ = {
        "model_tag": model_tag, "model": model, "effort": effort,
        "n_scored": len(scored), "n_parsed": len(vals), "unparsed": unparsed,
        "mean": round(sum(vals) / max(len(vals), 1), 3),
        "tok_in": tin, "tok_out": tout, "tok_reasoning": treason,
        "cost_usd": round(tin * pin + tout * pout, 4),
        "elapsed_sec": round(el, 1), "log": str(getattr(log, "location", "")),
    }
    print("LEVELSUM " + json.dumps(summ), flush=True)
    return summ


if __name__ == "__main__":
    model_tag = sys.argv[1]
    efforts = sys.argv[2].split(",")
    limit = int(os.getenv("SAMPLE_LIMIT", "0")) or None
    for e in efforts:
        run(model_tag, e, limit)
