#!/usr/bin/env python3
"""Saturation test: run Haiku 4.5 (via OpenRouter) on 100 random APPS problems
and grade generated code against the dataset's hidden test cases.

The model receives ONLY the problem statement (+ starter code / fn_name when
present). The hidden input/output test cases are never shown to the model;
they are used solely for local grading.
"""
import os, sys, json, re, random, subprocess, tempfile, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
N = 100
SEED = 42
MODEL = os.environ.get("EVAL_MODEL", "anthropic/claude-haiku-4.5")
TAG = os.environ.get("EVAL_TAG", "haiku-4.5")
RESULTS_PATH = os.path.join(HERE, f"results_{TAG}.json")
MAX_TOKENS = 3072
PER_TEST_TIMEOUT = 6           # seconds per test-case subprocess
MAX_TESTS_PER_PROBLEM = 20     # cap to bound runtime (noted in report)
WORKERS = 8
API_URL = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.environ["OPENROUTER_API_KEY"]

SYSTEM = (
    "You are an expert competitive programmer. You will be given a programming "
    "problem. Write a single, complete, correct Python 3 solution that solves "
    "it. Respond with ONLY the Python code inside one ``` ``` code block - no "
    "explanation, no prose before or after."
)

def build_user_prompt(question, starter_code, fn_name):
    parts = ["QUESTION:\n" + question.strip()]
    if starter_code and starter_code.strip():
        parts.append(starter_code.strip())
    if fn_name:
        parts.append(
            f"Use Call-Based format: implement the function named `{fn_name}`. "
            "It will be called with the provided arguments and its return value "
            "checked. Provide complete runnable Python 3 code."
        )
    else:
        parts.append(
            "Use Standard Input format: read input from standard input (stdin) "
            "and write the answer to standard output (stdout). Do not hardcode "
            "the sample inputs. Provide a complete runnable Python 3 program."
        )
    parts.append("ANSWER:")
    return "\n\n".join(parts)

CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

def extract_code(text):
    blocks = CODE_FENCE.findall(text or "")
    if blocks:
        return max(blocks, key=len).strip()   # largest fenced block
    return (text or "").strip()               # fall back to whole reply

def call_api(system, user, retries=3):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(API_URL, data=body, headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/saturation_test",
            "X-Title": "APPS saturation test",
        })
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read().decode())
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {}) or {}
            return content, usage, None
        except urllib.error.HTTPError as e:
            err_body = e.read().decode(errors="replace")[:300]
            last = f"HTTP {e.code}: {err_body}"
            if e.code in (402, 401, 403):      # credit/auth -> don't retry
                return None, {}, last
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(2 * (attempt + 1))
    return None, {}, last

# ---------- grading ----------
def norm_lines(s):
    lines = [ln.rstrip() for ln in s.replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return lines

def outputs_match(out, exp):
    no, ne = norm_lines(out), norm_lines(exp)
    if no == ne:
        return True
    if len(no) == len(ne):
        ok = True
        for a, b in zip(no, ne):
            ta, tb = a.split(), b.split()
            if ta == tb:
                continue
            if len(ta) != len(tb):
                ok = False; break
            for x, y in zip(ta, tb):
                if x == y:
                    continue
                try:
                    if abs(float(x) - float(y)) <= 1e-6:
                        continue
                except ValueError:
                    pass
                ok = False; break
            if not ok:
                break
        if ok:
            return True
    return out.split() == exp.split()

CALL_RUNNER = r'''
import sys, json
sol_path, tests_path, fn_name = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(sol_path).read()
tests = json.load(open(tests_path))
ns = {}
exec(compile(src, "<sol>", "exec"), ns)
fn = ns.get(fn_name)
if fn is None:
    # try a method on a class named Solution
    Sol = ns.get("Solution")
    if Sol is not None and hasattr(Sol, fn_name):
        fn = getattr(Sol(), fn_name)
if fn is None:
    print("NOFUNC"); sys.exit(3)
inputs, outputs = tests["inputs"], tests["outputs"]
npass = 0; total = 0
for args, exp in zip(inputs, outputs):
    total += 1
    try:
        got = fn(*args) if isinstance(args, list) else fn(args)
    except Exception:
        continue
    ok = (got == exp)
    if not ok:
        # APPS sometimes wraps the expected value in a list
        if isinstance(exp, list) and len(exp) == 1 and got == exp[0]:
            ok = True
        else:
            try:
                ok = json.dumps(got, sort_keys=True) == json.dumps(exp, sort_keys=True)
            except Exception:
                ok = False
    if ok:
        npass += 1
print(json.dumps({"pass": npass, "total": total}))
'''

def grade_stdin(code, io):
    inputs = io.get("inputs", [])[:MAX_TESTS_PER_PROBLEM]
    outputs = io.get("outputs", [])[:MAX_TESTS_PER_PROBLEM]
    n = min(len(inputs), len(outputs))
    if n == 0:
        return {"status": "failed_to_run", "reason": "no_tests", "n_pass": 0, "n_tests": 0}
    try:
        compile(code, "<sol>", "exec")
    except SyntaxError as e:
        return {"status": "failed_to_run", "reason": f"syntax_error: {e}", "n_pass": 0, "n_tests": n}
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); sol = f.name
    n_pass = n_wrong = n_err = n_timeout = 0
    try:
        for i in range(n):
            inp = inputs[i]
            exp = outputs[i]
            inp = inp if isinstance(inp, str) else "".join(inp) if isinstance(inp, list) else str(inp)
            exp = exp if isinstance(exp, str) else "".join(exp) if isinstance(exp, list) else str(exp)
            try:
                p = subprocess.run([sys.executable, sol], input=inp,
                                   capture_output=True, text=True,
                                   timeout=PER_TEST_TIMEOUT)
            except subprocess.TimeoutExpired:
                n_timeout += 1; continue
            if p.returncode != 0:
                n_err += 1; continue
            if outputs_match(p.stdout, exp):
                n_pass += 1
            else:
                n_wrong += 1
    finally:
        try: os.unlink(sol)
        except OSError: pass
    if n_pass == n:
        status = "passed_all"
    elif (n_pass + n_wrong) > 0:
        status = "wrong"
    else:
        status = "failed_to_run"
    return {"status": status, "reason": f"pass={n_pass} wrong={n_wrong} err={n_err} timeout={n_timeout}",
            "n_pass": n_pass, "n_tests": n}

def grade_call(code, io, fn_name):
    inputs = io.get("inputs", [])[:MAX_TESTS_PER_PROBLEM]
    outputs = io.get("outputs", [])[:MAX_TESTS_PER_PROBLEM]
    n = min(len(inputs), len(outputs))
    if n == 0:
        return {"status": "failed_to_run", "reason": "no_tests", "n_pass": 0, "n_tests": 0}
    try:
        compile(code, "<sol>", "exec")
    except SyntaxError as e:
        return {"status": "failed_to_run", "reason": f"syntax_error: {e}", "n_pass": 0, "n_tests": n}
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); sol = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"inputs": inputs[:n], "outputs": outputs[:n]}, f); tf = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(CALL_RUNNER); rf = f.name
    try:
        try:
            p = subprocess.run([sys.executable, rf, sol, tf, fn_name],
                               capture_output=True, text=True,
                               timeout=PER_TEST_TIMEOUT * 4)
        except subprocess.TimeoutExpired:
            return {"status": "failed_to_run", "reason": "timeout", "n_pass": 0, "n_tests": n}
        if p.returncode != 0 or not p.stdout.strip():
            return {"status": "failed_to_run", "reason": f"runtime: {p.stderr[:120]}", "n_pass": 0, "n_tests": n}
        try:
            res = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            return {"status": "failed_to_run", "reason": "bad_runner_output", "n_pass": 0, "n_tests": n}
        n_pass = res["pass"]
        status = "passed_all" if n_pass == n else "wrong"
        return {"status": status, "reason": f"pass={n_pass}/{n}", "n_pass": n_pass, "n_tests": n}
    finally:
        for x in (sol, tf, rf):
            try: os.unlink(x)
            except OSError: pass

# ---------- main ----------
def load_sample():
    from datasets import load_dataset
    ds = load_dataset("codeparrot/apps", split="test",
                      trust_remote_code=True, streaming=True)
    rows = []
    for r in ds:
        rows.append({k: r[k] for k in ("problem_id", "question", "input_output",
                                       "difficulty", "starter_code")})
    rnd = random.Random(SEED)
    idx = rnd.sample(range(len(rows)), N)
    return [rows[i] for i in idx]

def process(item):
    io = {}
    fn_name = None
    raw_io = item.get("input_output") or ""
    if raw_io:
        try:
            io = json.loads(raw_io)
            fn_name = io.get("fn_name")
        except Exception:
            io = {}
    user = build_user_prompt(item["question"], item.get("starter_code"), fn_name)
    content, usage, err = call_api(SYSTEM, user)
    rec = {"problem_id": item["problem_id"], "difficulty": item["difficulty"],
           "mode": "call" if fn_name else "stdin",
           "prompt_tokens": usage.get("prompt_tokens", 0),
           "completion_tokens": usage.get("completion_tokens", 0)}
    if err and content is None:
        rec.update(status="api_error", reason=err, n_pass=0, n_tests=0, code="")
        return rec
    code = extract_code(content)
    rec["code"] = code
    if not code.strip():
        rec.update(status="failed_to_run", reason="no_code", n_pass=0, n_tests=0)
        return rec
    if io and not io.get("inputs"):
        rec.update(status="skipped", reason="no_tests", n_pass=0, n_tests=0)
        return rec
    g = grade_call(code, io, fn_name) if fn_name else grade_stdin(code, io)
    rec.update(g)
    return rec

def main():
    print(f"[setup] loading + sampling {N} problems (seed={SEED}) ...", flush=True)
    sample = load_sample()
    modes = {}
    for s in sample:
        try:
            d = json.loads(s["input_output"] or "{}")
            modes["call" if "fn_name" in d else "stdin"] = modes.get("call" if "fn_name" in d else "stdin", 0) + 1
        except Exception:
            modes["bad"] = modes.get("bad", 0) + 1
    print(f"[setup] sampled. modes={modes}. running with {WORKERS} workers ...", flush=True)

    results = []
    counts = {"passed_all": 0, "wrong": 0, "failed_to_run": 0, "api_error": 0, "skipped": 0}
    tok_in = tok_out = 0
    done = 0
    lock = threading.Lock()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process, item): item for item in sample}
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                results.append(rec)
                counts[rec["status"]] = counts.get(rec["status"], 0) + 1
                tok_in += rec.get("prompt_tokens", 0)
                tok_out += rec.get("completion_tokens", 0)
                done += 1
                if done % 5 == 0 or done == N:
                    el = time.time() - t0
                    cost = tok_in / 1e6 * 1.0 + tok_out / 1e6 * 5.0
                    print(f"[{done:3d}/{N}] passed_all={counts['passed_all']:3d} "
                          f"wrong={counts['wrong']:3d} failed_to_run={counts['failed_to_run']:3d} "
                          f"api_error={counts['api_error']} skipped={counts['skipped']} "
                          f"| ~${cost:0.3f} | {el:0.0f}s", flush=True)
            # incremental save
            with open(RESULTS_PATH, "w") as f:
                json.dump({"counts": counts, "results": results}, f, indent=2)

    # final breakdown
    by_diff = {}
    for r in results:
        d = r["difficulty"]; by_diff.setdefault(d, {"passed_all": 0, "total": 0})
        by_diff[d]["total"] += 1
        if r["status"] == "passed_all":
            by_diff[d]["passed_all"] += 1
    cost = tok_in / 1e6 * 1.0 + tok_out / 1e6 * 5.0
    summary = {"counts": counts, "by_difficulty": by_diff,
               "tokens": {"input": tok_in, "output": tok_out},
               "est_cost_usd": round(cost, 4),
               "graded_max_tests_per_problem": MAX_TESTS_PER_PROBLEM,
               "seed": SEED, "model": MODEL}
    with open(RESULTS_PATH, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print("\n===== FINAL =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nPASSED ALL TEST CASES: {counts['passed_all']}/{N}", flush=True)
    print(f"WRONG (ran but failed >=1 case): {counts['wrong']}/{N}", flush=True)
    print(f"FAILED TO RUN (no code / syntax / crashed-or-timed-out on all): "
          f"{counts['failed_to_run']}/{N}", flush=True)
    if counts["api_error"]:
        print(f"API ERRORS: {counts['api_error']}/{N}", flush=True)

if __name__ == "__main__":
    main()
