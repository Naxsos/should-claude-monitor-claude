"""Re-generate only the failed (model, problem) rows in solutions.jsonl (the 402
credit casualties), in place. Reuses gen_solutions.call/extract_code/MODELS."""
import json, os
from concurrent.futures import ThreadPoolExecutor, as_completed
import gen_solutions as g

HERE = g.HERE
probs = {p["problem_id"]: p["question"] for p in json.load(open(os.path.join(HERE, "problems.json")))}
rows = [json.loads(l) for l in open(os.path.join(HERE, "solutions.jsonl"))]
fails = [(i, r) for i, r in enumerate(rows) if not r["ok"]]
print(f"re-running {len(fails)} failed rows", flush=True)


def redo(i, r):
    slug, fam, eff = g.MODELS[r["model"]]
    res, err = g.call(slug, eff, probs[r["problem_id"]])
    if res:
        code = g.extract_code(res["text"])
        return i, {"model": r["model"], "family": fam, "effort": eff, "problem_id": r["problem_id"],
                   "code": code, "ok": bool(code.strip()), "err": None if code.strip() else "empty",
                   "tok_in": res["in"], "tok_out": res["out"], "code_len": len(code)}, None
    return i, None, err


fixed = 0
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = [ex.submit(redo, i, r) for i, r in fails]
    for f in as_completed(futs):
        i, newrow, err = f.result()
        if newrow and newrow["ok"]:
            rows[i] = newrow; fixed += 1
        else:
            print(f"  still failing: {rows[i]['model']} pid{rows[i]['problem_id']} {err}", flush=True)
with open(os.path.join(HERE, "solutions.jsonl"), "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
ok = sum(1 for r in rows if r["ok"])
print(f"fixed {fixed}/{len(fails)} | solutions.jsonl now {ok}/{len(rows)} ok", flush=True)
