"""Smoke: 9 models x 3 APPS problems, 1-shot generation, check format compliance.
Verifies the response fits the ```python\\n{code}\\n``` spec. Captures RAW output
so we can tell a proper fenced block from raw code. No iteration."""
import os, re, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

KEY = os.environ["OPENROUTER_API_KEY"]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "thursday_experiments_zu"))
from prompts import SELFREC_GEN_SYS as SYS, SELFREC_GEN_USER as USER  # noqa: E402
import gen_solutions as g  # reuse MODELS, reasoning_for  # noqa: E402

PROBS = json.load(open(os.path.join(HERE, "problems.json")))[:3]
PY_FENCE = re.compile(r"```python\s*(.*?)```", re.S | re.I)
ANY_FENCE = re.compile(r"```\s*(.*?)```", re.S)


def call(slug, eff, q):
    body = {"model": slug, "messages": [{"role": "system", "content": SYS},
            {"role": "user", "content": USER.format(q=q)}],
            "max_tokens": 16000, "temperature": 0.0, "reasoning": g.reasoning_for(eff)}
    try:
        d = requests.post("https://openrouter.ai/api/v1/chat/completions",
                          headers={"Authorization": f"Bearer {KEY}"}, json=body, timeout=300).json()
        if d.get("error"):
            return None, f"err:{d['error'].get('code')}"
        return d["choices"][0]["message"]["content"], None
    except Exception as e:
        return None, f"exc:{str(e)[:60]}"


def classify(text):
    if text is None:
        return "NO_RESPONSE"
    if PY_FENCE.search(text):
        return "python_fence"       # fits spec exactly
    if ANY_FENCE.search(text):
        return "bare_fence"         # ``` without 'python'
    return "no_fence"               # raw code / prose, no block


jobs = [(tag, slug, fam, eff, p) for tag, (slug, fam, eff) in g.MODELS.items() for p in PROBS]
rows = []
with ThreadPoolExecutor(max_workers=9) as ex:
    futs = {ex.submit(call, slug, eff, p["question"]): (tag, p["problem_id"]) for tag, slug, fam, eff, p in jobs}
    for f in as_completed(futs):
        tag, pid = futs[f]
        txt, err = f.result()
        rows.append({"model": tag, "pid": pid, "status": classify(txt),
                     "err": err, "resp_len": len(txt) if txt else 0,
                     "head": (txt[:60].replace("\n", "\\n") if txt else "")})

print(f"{'model':22} {'spec-fit (python_fence)':>22}  detail")
for tag in g.MODELS:
    r = [x for x in rows if x["model"] == tag]
    fit = sum(1 for x in r if x["status"] == "python_fence")
    other = [x["status"] for x in r if x["status"] != "python_fence"]
    print(f"{tag:22} {f'{fit}/3':>22}  {('' if not other else 'NON-SPEC: '+str(other))}")
print()
# show any non-compliant heads
bad = [x for x in rows if x["status"] != "python_fence"]
if bad:
    print("--- non-compliant samples (first 60 chars) ---")
    for x in bad:
        print(f"  {x['model']:20} {x['status']:12} {x['err'] or ''} | {x['head']}")
else:
    print("ALL responses fit the ```python ... ``` spec.")
