#!/usr/bin/env python3
"""Rerun only the `failed_to_run` problems from a prior saturation run, with a
larger MAX_TOKENS, to test the truncation hypothesis (unclosed code fence ->
syntax_error line 1). Reuses run_eval's process()/grading verbatim; only the
token cap changes. Writes rerun_failed<N>_<MAXTOK>.json and prints before/after.
"""
import os, sys, json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_eval as R  # reads MODEL/KEY from env at import time

NEW_MAX_TOKENS = int(os.environ.get("NEW_MAX_TOKENS", "6144"))
SRC = os.environ.get("SRC_RESULTS", os.path.join(HERE, "results_deepseek-v3.2-exp.json"))
R.MAX_TOKENS = NEW_MAX_TOKENS  # call_api() reads this module global

old = json.load(open(SRC))
old_by_id = {r["problem_id"]: r for r in old["results"]}
failed_ids = {r["problem_id"] for r in old["results"] if r["status"] == "failed_to_run"}
print(f"[setup] model={R.MODEL} | MAX_TOKENS {3072} -> {R.MAX_TOKENS} | rerunning {len(failed_ids)} failed_to_run pids", flush=True)

sample = R.load_sample()  # same SEED/N => identical 100; filter to the failures
items = [it for it in sample if it["problem_id"] in failed_ids]
print(f"[setup] matched {len(items)} / {len(failed_ids)} items from the seed-{R.SEED} sample", flush=True)

new_results = []
with ThreadPoolExecutor(max_workers=R.WORKERS) as ex:
    futs = {ex.submit(R.process, it): it for it in items}
    for fut in as_completed(futs):
        rec = fut.result()
        new_results.append(rec)
        o = old_by_id[rec["problem_id"]]
        print(f"  pid={rec['problem_id']:>5} {rec['difficulty']:12} "
              f"{o['status']:14} -> {rec['status']:14} "
              f"ctoks {o['completion_tokens']:>5}->{rec['completion_tokens']:>5}  "
              f"{str(rec.get('reason',''))[:55]}", flush=True)

print("\n===== RERUN SUMMARY =====", flush=True)
print("old status dist:", dict(Counter(old_by_id[r['problem_id']]['status'] for r in new_results)))
print("new status dist:", dict(Counter(r["status"] for r in new_results)))
recovered = [r for r in new_results if r["status"] == "passed_all"]
ran_now = [r for r in new_results if r["status"] in ("passed_all", "wrong")]
still_cap = [r for r in new_results if r["completion_tokens"] >= R.MAX_TOKENS]
print(f"now passed_all: {len(recovered)}/{len(new_results)}")
print(f"now runs (passed_all+wrong): {len(ran_now)}/{len(new_results)}")
print(f"still hitting the {R.MAX_TOKENS} cap: {len(still_cap)}")
out = os.path.join(HERE, f"rerun_failed{len(new_results)}_{R.MAX_TOKENS}.json")
json.dump({"new_max_tokens": R.MAX_TOKENS, "results": new_results}, open(out, "w"), indent=2)
print("wrote", out, flush=True)
