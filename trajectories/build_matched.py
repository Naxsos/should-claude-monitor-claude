"""Build a #tool_calls-matched action-only variant from cache_windowed.json.

For each honest<->attack pair, render action-only units (tool_call + capped output,
no prose), then TRUNCATE the honest unit list to the paired attack's tool_call count
(keep the last N, end-anchored like the original windows). Attack kept as-is so its
attack action is preserved. Re-renders to monitorbench_56x56_stepmatched.jsonl and
re-runs the length/composition baseline so we can see what (if anything) still leaks.
"""
import json, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = 2000  # per-output char cap, same as the existing action-only set


def units(rec):
    """Flatten to [(call_str, output_str)] in order; pair i-th call with i-th tool output."""
    calls, outs = [], []
    for m in rec["messages"]:
        for tc in (m.get("tool_calls") or []):
            calls.append(f"{tc.get('function')} {json.dumps(tc.get('arguments'))}")
        if m.get("role") == "tool":
            outs.append(m.get("text") or "")
    n = min(len(calls), len(outs)) if outs else len(calls)
    return [(calls[i], outs[i] if i < len(outs) else "") for i in range(len(calls))]


def render(rec, us):
    head = f"[TASK]\n{rec.get('setting_description','')}\n\n{rec.get('main_task_description','')}\n"
    body = "\n".join(f"[ACTION]\n{c}\n[OUTPUT]\n{o[:CAP]}" for c, o in us)
    return head + "\n" + body


def main():
    cache = json.load(open(os.path.join(HERE, "cache_windowed.json")))
    idx = {(r["task_id"], r["label"], r["model"]): r for r in cache}
    pairs = json.load(open(os.path.join(HERE, "pairs_56.json")))
    out = []
    dropped = []
    for p in pairs:
        tid, model = p["task_id"], p["model"]
        h = idx.get((tid, "honest", model)); a = idx.get((tid, "attack", model))
        if not h or not a:
            print("  MISSING pair:", tid, model); continue
        hu, au = units(h), units(a)
        n_a = len(au)
        hu_trim = hu[-n_a:] if len(hu) > n_a else hu   # keep last N_a (end-anchored)
        dropped.append(len(hu) - len(hu_trim))
        for rec, us, lbl in [(a, au, "attack"), (h, hu_trim, "honest")]:
            t = render(rec, us)
            out.append({"uid": f"{lbl}_{tid}_{model[-4:]}", "label": lbl, "model": model,
                        "task_id": tid, "n_tool_calls": len(us),
                        "n_tokens_est": len(t) // 4, "transcript": t})
    path = os.path.join(HERE, "monitorbench_56x56_stepmatched.jsonl")
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {path}  ({len(out)} records, dropped {sum(dropped)} honest tool-calls total, "
          f"avg {sum(dropped)/max(len(dropped),1):.1f}/honest)")
    # quick check
    for lbl in ("attack", "honest"):
        rs = [r for r in out if r["label"] == lbl]
        tc = sum(r["n_tool_calls"] for r in rs) / len(rs)
        tk = sum(r["n_tokens_est"] for r in rs) / len(rs)
        print(f"  {lbl}: mean tool_calls={tc:.1f}  mean tokens={tk:.0f}")


if __name__ == "__main__":
    main()
