"""Self-recognition experiment — STAGE 2: family recognition.

For each of the 9 judge models, show it 200 solutions and ask 0-1 how likely the
code was written by a model from its OWN family/developer:
  - 100 sampled uniformly from its own family's 300 solutions (self + siblings)
  - 100 sampled uniformly from the other 6 models' 600 solutions
Each judged at the model's own reasoning effort (high effort kept ON).

Reads selfrec/solutions.jsonl ; writes selfrec/recognition.jsonl.
Smoke: SAMPLE_LIMIT=5 python3 recognize.py   (5 own + 5 other per judge)
"""
import os, re, json, time, random
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

KEY = os.environ["OPENROUTER_API_KEY"]
BASE = "https://openrouter.ai/api/v1/chat/completions"
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
WORKERS = 8
MAX_TOKENS = 8000
N_PER_SIDE = int(os.getenv("SAMPLE_LIMIT", "0")) or 100   # per own / other side

MODELS = {
    "claude-sonnet-4.6":     ("anthropic/claude-sonnet-4.6",      "claude", "high"),
    "claude-sonnet-4.5":     ("anthropic/claude-sonnet-4.5",      "claude", None),
    "claude-haiku-4.5":      ("anthropic/claude-haiku-4.5",       "claude", None),
    "gemini-3.5-flash":      ("google/gemini-3.5-flash",          "gemini", "high"),
    "gemini-3-flash":        ("google/gemini-3-flash-preview",    "gemini", "high"),
    "gemini-3.1-flash-lite": ("google/gemini-3.1-flash-lite",     "gemini", "high"),
    "gpt-5.2":               ("openai/gpt-5.2",                   "gpt",    None),
    "gpt-5.4-nano":          ("openai/gpt-5.4-nano",              "gpt",    "low"),
    "gpt-5-nano":            ("openai/gpt-5-nano",                "gpt",    "low"),
}
FAMILY_LABEL = {"claude": "Anthropic (Claude)", "gemini": "Google (Gemini)", "gpt": "OpenAI (GPT)"}

import sys as _sys
_sys.path.insert(0, os.path.dirname(HERE))   # repo root holds prompts_old.py (superseded single-item)
from prompts_old import SELFREC_SINGLE_SYS as SYS, SELFREC_SINGLE_USER as USER, developer_from_slug  # noqa: E402

_TAG = re.compile(r"<SCORE>\s*(0?\.\d+|1(?:\.0+)?|0|1)\s*</SCORE>", re.I)
_FLOAT = re.compile(r"(?<![\d.])(0?\.\d+|1\.0+|0|1)(?![\d.])")


def parse_score(t):
    if not t:
        return None
    m = _TAG.search(t)
    v = m.group(1) if m else (_FLOAT.findall(t) or [None])[-1]
    try:
        v = float(v); return v if 0.0 <= v <= 1.0 else None
    except (TypeError, ValueError):
        return None


def reasoning_for(effort):
    return {"enabled": False} if effort in (None, "none") else {"effort": effort}


def call(slug, effort, code, developer, retries=4):
    body = {"model": slug, "messages": [{"role": "system", "content": SYS.format(developer=developer)},
            {"role": "user", "content": USER.format(code=code)}],
            "max_tokens": MAX_TOKENS, "temperature": 0.0, "reasoning": reasoning_for(effort)}
    for a in range(retries):
        try:
            r = requests.post(BASE, headers={"Authorization": f"Bearer {KEY}"}, json=body, timeout=300)
            d = r.json()
            if d.get("error"):
                if a == retries - 1:
                    return None, f"err:{d['error'].get('code')}"
                time.sleep(2 * (a + 1)); continue
            return d["choices"][0]["message"]["content"], None
        except Exception as e:
            if a == retries - 1:
                return None, f"exc:{str(e)[:60]}"
            time.sleep(2 * (a + 1))
    return None, "exhausted"


def main():
    sols = [json.loads(l) for l in open(os.path.join(HERE, "solutions.jsonl"))
            if json.loads(l)["ok"] and json.loads(l)["code"].strip()]
    by_fam = {}
    for s in sols:
        by_fam.setdefault(s["family"], []).append(s)
    print(f"loaded {len(sols)} solutions | by family: {{ {', '.join(f'{k}:{len(v)}' for k,v in by_fam.items())} }}", flush=True)

    jobs = []
    for ji, (tag, (slug, fam, eff)) in enumerate(MODELS.items()):
        rng = random.Random(SEED + ji)
        own = list(by_fam[fam]); other = [s for f, v in by_fam.items() if f != fam for s in v]
        own_s = rng.sample(own, min(N_PER_SIDE, len(own)))
        oth_s = rng.sample(other, min(N_PER_SIDE, len(other)))
        for s in own_s:
            jobs.append((tag, slug, fam, eff, s, True))
        for s in oth_s:
            jobs.append((tag, slug, fam, eff, s, False))
    print(f"=== recognition | {len(MODELS)} judges x {2*N_PER_SIDE} = {len(jobs)} jobs ===", flush=True)

    out = os.path.join(HERE, "recognition_smoke.jsonl" if N_PER_SIDE < 100 else "recognition.jsonl")
    fh = open(out, "w"); done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(call, slug, eff, s["code"], developer_from_slug(fam)): (tag, fam, s, own)
                for tag, slug, fam, eff, s, own in jobs}
        for f in as_completed(futs):
            tag, fam, s, own = futs[f]
            txt, err = f.result()
            sc = parse_score(txt) if txt else None
            fh.write(json.dumps({"judge": tag, "judge_family": fam, "item_model": s["model"],
                     "item_family": s["family"], "is_own": own, "score": sc,
                     "err": err}) + "\n"); fh.flush()
            done += 1
            if err or done % 100 == 0:
                print(f"  [{done}/{len(jobs)}] {tag} {'OK' if not err else 'FAIL '+str(err)}", flush=True)
    fh.close()
    print(f"wrote {out} ({done} rows)", flush=True)


if __name__ == "__main__":
    main()
