"""Self-recognition experiment — STAGE 1: generation.

9 models (3 families x 3 capability tiers) each generate a 1-shot Python solution
to the SAME 100 APPS problems (random.seed(42)). Correctness is NOT required.

Writes selfrec/solutions.jsonl: one row per (model, problem) with the extracted code.
Direct OpenRouter Chat Completions; concurrent; retries. Per-model reasoning config.

Smoke:  SAMPLE_LIMIT=1 python3 gen_solutions.py
Full:    python3 gen_solutions.py
"""
import os, re, json, time, random, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

KEY = os.environ["OPENROUTER_API_KEY"]
BASE = "https://openrouter.ai/api/v1/chat/completions"
HERE = os.path.dirname(os.path.abspath(__file__))
N_PROBLEMS = 100
SEED = 42
WORKERS = 8
MAX_TOKENS = 16000         # room for code + heavy reasoning on high-effort models (smoke hit 8k cap)
LIMIT = int(os.getenv("SAMPLE_LIMIT", "0")) or None

# tag -> (slug, family, effort)   effort None/"none" => no extended thinking
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

sys.path.insert(0, os.path.join(os.path.dirname(HERE), "thursday_experiments_zu"))
from prompts import SELFREC_GEN_SYS as SYS, SELFREC_GEN_USER as USER  # noqa: E402

_CODE = re.compile(r"```(?:python)?\s*(.*?)```", re.S | re.I)


def extract_code(text):
    if not text:
        return ""
    blocks = _CODE.findall(text)
    return max(blocks, key=len).strip() if blocks else text.strip()


def reasoning_for(effort):
    return {"enabled": False} if effort in (None, "none") else {"effort": effort}


def load_problems():
    cache = os.path.join(HERE, "problems.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
    from datasets import load_dataset
    ds = load_dataset("codeparrot/apps", split="test", streaming=True, trust_remote_code=True)
    rows = list(ds.take(5000))
    idx = random.Random(SEED).sample(range(len(rows)), N_PROBLEMS)
    probs = [{"problem_id": int(rows[i].get("problem_id", i)), "question": rows[i]["question"]} for i in idx]
    json.dump(probs, open(cache, "w"))
    return probs


def call(slug, effort, q, retries=4):
    body = {"model": slug, "messages": [{"role": "system", "content": SYS},
            {"role": "user", "content": USER.format(q=q)}],
            "max_tokens": MAX_TOKENS, "temperature": 0.0, "reasoning": reasoning_for(effort)}
    for a in range(retries):
        try:
            r = requests.post(BASE, headers={"Authorization": f"Bearer {KEY}"}, json=body, timeout=300)
            d = r.json()
            if d.get("error"):
                if a == retries - 1:
                    return None, f"err:{d['error'].get('code')}:{str(d['error'].get('message'))[:80]}"
                time.sleep(2 * (a + 1)); continue
            msg = d["choices"][0]["message"]["content"]
            u = d.get("usage", {})
            return {"text": msg, "in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}, None
        except Exception as e:
            if a == retries - 1:
                return None, f"exc:{str(e)[:80]}"
            time.sleep(2 * (a + 1))
    return None, "exhausted"


def main():
    probs = load_problems()
    if LIMIT:
        probs = probs[:LIMIT]
    jobs = [(tag, slug, fam, eff, p) for tag, (slug, fam, eff) in MODELS.items() for p in probs]
    print(f"=== generation | {len(MODELS)} models x {len(probs)} problems = {len(jobs)} jobs ===", flush=True)
    out_path = os.path.join(HERE, "solutions_smoke.jsonl" if LIMIT else "solutions.jsonl")
    done = 0
    fh = open(out_path, "w")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(call, slug, eff, p["question"]): (tag, fam, eff, p) for tag, slug, fam, eff, p in jobs}
        for f in as_completed(futs):
            tag, fam, eff, p = futs[f]
            res, err = f.result()
            rec = {"model": tag, "family": fam, "effort": eff, "problem_id": p["problem_id"],
                   "code": extract_code(res["text"]) if res else "", "ok": err is None,
                   "err": err, "tok_in": res["in"] if res else 0, "tok_out": res["out"] if res else 0,
                   "code_len": len(extract_code(res["text"])) if res else 0}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            done += 1
            if err or done % 50 == 0:
                print(f"  [{done}/{len(jobs)}] {tag} pid{p['problem_id']} {'OK' if not err else 'FAIL '+err}", flush=True)
    fh.close()
    print(f"wrote {out_path} ({done} rows)", flush=True)


if __name__ == "__main__":
    main()
