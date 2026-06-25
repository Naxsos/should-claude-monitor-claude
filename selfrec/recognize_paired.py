"""Self-recognition STAGE 2 (PAIRED / 2AFC variant).

For each of the 9 judge models, show it PAIRS of solutions (one from its OWN
family, one from a DIFFERENT family) and ask which — A or B — is its own family,
plus a 0-1 confidence. Own-family solution position (A/B) is randomized & recorded
to control for position bias.

Pairs per judge = N_PAIRS (default 100). own sampled from own family's solutions
(self+siblings), other sampled from the other 6 models' solutions; seeded.

Reads selfrec/solutions.jsonl ; writes selfrec/recognition_paired.jsonl.
Smoke: SAMPLE_LIMIT=5 python3 recognize_paired.py
"""
import os, re, json, time, random, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

KEY = os.environ["OPENROUTER_API_KEY"]
BASE = "https://openrouter.ai/api/v1/chat/completions"
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
WORKERS = 8
MAX_TOKENS = 8000
N_PAIRS = int(os.getenv("SAMPLE_LIMIT", "0")) or 100

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
from prompts import SELFREC_PAIRED_SYS as SYS, SELFREC_PAIRED_USER as USER, developer_from_slug  # noqa: E402

_CH = re.compile(r"<CHOICE>\s*([AB])\s*</CHOICE>", re.I)
_CF = re.compile(r"<CONFIDENCE>\s*(0?\.\d+|1(?:\.0+)?|0|1)\s*</CONFIDENCE>", re.I)


def parse(t):
    if not t:
        return None, None
    mc = _CH.search(t); mf = _CF.search(t)
    choice = mc.group(1).upper() if mc else None
    conf = None
    if mf:
        try:
            v = float(mf.group(1)); conf = v if 0 <= v <= 1 else None
        except ValueError:
            pass
    return choice, conf


def reasoning_for(effort):
    return {"enabled": False} if effort in (None, "none") else {"effort": effort}


def call(slug, effort, a, b, developer, retries=4):
    body = {"model": slug, "messages": [{"role": "system", "content": SYS.format(developer=developer)},
            {"role": "user", "content": USER.format(a=a, b=b)}],
            "max_tokens": MAX_TOKENS, "temperature": 0.0, "reasoning": reasoning_for(effort)}
    for k in range(retries):
        try:
            r = requests.post(BASE, headers={"Authorization": f"Bearer {KEY}"}, json=body, timeout=300)
            d = r.json()
            if d.get("error"):
                if k == retries - 1:
                    return None, f"err:{d['error'].get('code')}"
                time.sleep(2 * (k + 1)); continue
            return d["choices"][0]["message"]["content"], None
        except Exception as e:
            if k == retries - 1:
                return None, f"exc:{str(e)[:60]}"
            time.sleep(2 * (k + 1))
    return None, "exhausted"


def main():
    sols = [json.loads(l) for l in open(os.path.join(HERE, "solutions.jsonl"))
            if json.loads(l)["ok"] and json.loads(l)["code"].strip()]
    by_fam = {}
    for s in sols:
        by_fam.setdefault(s["family"], []).append(s)
    print(f"loaded {len(sols)} | by family: {{ {', '.join(f'{k}:{len(v)}' for k,v in by_fam.items())} }}", flush=True)

    jobs = []
    for ji, (tag, (slug, fam, eff)) in enumerate(MODELS.items()):
        rng = random.Random(SEED + ji)
        own = list(by_fam[fam]); other = [s for f, v in by_fam.items() if f != fam for s in v]
        own_s = rng.sample(own, min(N_PAIRS, len(own)))
        oth_s = rng.sample(other, min(N_PAIRS, len(other)))
        for o, x in zip(own_s, oth_s):
            own_pos = rng.choice("AB")                       # randomize own-family position
            a, b = (o, x) if own_pos == "A" else (x, o)
            jobs.append((tag, slug, fam, eff, a, b, own_pos, o["model"], x["model"]))
    print(f"=== paired recognition | {len(MODELS)} judges x {N_PAIRS} pairs = {len(jobs)} jobs ===", flush=True)

    out = os.path.join(HERE, "recognition_paired_smoke.jsonl" if N_PAIRS < 100 else "recognition_paired.jsonl")
    fh = open(out, "w"); done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(call, slug, eff, a["code"], b["code"], developer_from_slug(fam)): (tag, fam, own_pos, om, xm, a["model"], b["model"])
                for tag, slug, fam, eff, a, b, own_pos, om, xm in jobs}
        for f in as_completed(futs):
            tag, fam, own_pos, om, xm, am, bm = futs[f]
            txt, err = f.result()
            choice, conf = parse(txt) if txt else (None, None)
            correct = (choice == own_pos) if choice else None
            fh.write(json.dumps({"judge": tag, "judge_family": fam, "own_pos": own_pos,
                     "choice": choice, "correct": correct, "confidence": conf,
                     "own_model": om, "other_model": xm,
                     "a_model": am, "b_model": bm,  # which model's code sat at A / B (provenance)
                     "own_is_self": (om == tag),  # split self-recognition vs sibling-recognition in analysis
                     "err": err}) + "\n"); fh.flush()
            done += 1
            if err or done % 100 == 0:
                print(f"  [{done}/{len(jobs)}] {tag} {'OK' if not err else 'FAIL '+str(err)}", flush=True)
    fh.close()
    print(f"wrote {out} ({done} rows)", flush=True)


if __name__ == "__main__":
    main()
