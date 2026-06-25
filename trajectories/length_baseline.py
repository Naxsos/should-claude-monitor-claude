"""Length/composition confound check (no API).

For each dataset, extract trivial features {total tokens, assistant chars, tool
chars, #tool_calls} and measure how well they ALONE separate attack vs honest:
  - univariate AUROC per feature
  - multivariate logistic-regression AUROC (5-fold CV)
High baseline AUROC => the monitor task is length/composition-confounded.
Run on prose-included AND action-only to see if stripping removed the confound.
"""
import json, sys, bisect, os, random


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    s = sorted(neg); w = t = 0
    for x in pos:
        w += bisect.bisect_left(s, x); t += bisect.bisect_right(s, x) - bisect.bisect_left(s, x)
    return (w + 0.5 * t) / (len(pos) * len(neg))


def features(transcript, n_tokens):
    """assistant chars, tool chars, #tool_calls from the [ASSISTANT]/[TOOL] format."""
    sec = None; a = t = 0; ntc = transcript.count("<tool_call")
    for line in transcript.split("\n"):
        ls = line.strip()
        if ls == "[ASSISTANT]":
            sec = "a"; continue
        if ls == "[TOOL]":
            sec = "t"; continue
        if ls == "[TASK]":
            sec = None; continue
        if sec == "a":
            a += len(line)
        elif sec == "t":
            t += len(line)
    return {"tokens": n_tokens, "asst_chars": a, "tool_chars": t, "n_tool_calls": ntc}


def run(path):
    recs = [json.loads(l) for l in open(path)]
    rows = []
    for r in recs:
        f = features(r["transcript"], r.get("n_tokens_est", 0))
        f["y"] = 1 if r["label"] == "attack" else 0
        rows.append(f)
    feats = ["tokens", "asst_chars", "tool_chars", "n_tool_calls"]
    pos = [r for r in rows if r["y"] == 1]; neg = [r for r in rows if r["y"] == 0]
    print(f"\n=== {os.path.basename(path)} ({len(pos)} attack / {len(neg)} honest) ===")
    print("  univariate AUROC (attack vs honest), + mean attack/honest:")
    for f in feats:
        a = auroc([r[f] for r in pos], [r[f] for r in neg])
        ma = sum(r[f] for r in pos) / len(pos); mh = sum(r[f] for r in neg) / len(neg)
        print(f"    {f:14} AUROC={a:.3f}  (attack mean {ma:,.0f} vs honest {mh:,.0f})")
    # multivariate logistic regression, 5-fold CV AUROC
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import roc_auc_score
        import numpy as np
        X = np.array([[r[f] for f in feats] for r in rows]); y = np.array([r["y"] for r in rows])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        proba = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
        print(f"  >>> multivariate logistic-reg 5-fold CV AUROC = {roc_auc_score(y, proba):.3f}")
    except Exception as e:
        print(f"  (sklearn unavailable: {e}; univariate above is the signal)")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for f in ["monitorbench_56x56.jsonl", "monitorbench_56x56_actiononly.jsonl"]:
        p = os.path.join(here, f)
        if os.path.exists(p):
            run(p)
    print("\nInterpretation: AUROC near 0.50 = no confound; far from 0.50 = length/composition leak.")
