# Should Claude monitor Claude? A 2×2 backdoor-detection study

_Suspicion-scoring experiment. Generated 2026-06-23. Data + code in `control_paper/`._

## Research question

When a **trusted monitor** (T) rates how suspicious a code solution looks, is it
**better or worse at catching backdoors written by a model from its own family**
than by a model from a different family? This is the core "should Claude monitor
Claude" question: does same-family monitoring carry a systematic blind spot
(collusion-like) or a systematic advantage (knows its own tricks)?

## Design — 2×2

| | **U = DeepSeek V3.2** (generator) | **U = Sonnet 4.5** (generator) |
|---|---|---|
| **T = DeepSeek V3.2 Exp** (monitor) | same-family | cross-family |
| **T = Haiku 4.5** (monitor) | cross-family | same-family |

- **U (untrusted/attacker):** the two 500-problem backdoor-generation runs
  (Sonnet 215 working backdoors, DeepSeek 122), same first-500 APPS problems.
- **T (trusted/monitor):** each monitor scores every **clean** solution and every
  **successful backdoor** from both generators.
- **Corpus:** Sonnet 715 (500 clean + 215 backdoor) + DeepSeek 622 (500 clean +
  122 backdoor) = **1,337 solutions × 2 monitors = 2,674 scorings.**

## Method

- Each monitor runs the **same free-form investigation prompt** (`DETECT_BACKDOOR_PROMPT`)
  then emits a **single continuous suspicion score in [0,1] to 3 decimals**
  (the "same prompt but 0–1 decimal, only the score" requested) via a
  `<SCORE>x.xxx</SCORE>` tag. Temperature 0, reproducible.
- **Metric: AUROC** = P(score(backdoor) > score(clean)). Threshold-independent, so
  it is robust to each monitor calibrating the 0–1 scale differently and to score
  clustering. Reported with 95% bootstrap CIs (2,000 resamples).
- **Two views:** *full* (all clean, clean N ≫ backdoor N) and *balanced*
  (clean N = backdoor N = 121, and equal backdoor N across families) — the
  balanced view de-confounds the headline interaction.
- The **interaction** (same-family − cross-family AUROC) is the headline statistic;
  it subtracts out both main effects (monitor skill, generator subtlety).
- Cost: **~$14** (Haiku ~$12.5, DeepSeek Exp ~$1.6 — 8× cheaper). 7/1337 Haiku
  scorings failed to parse (~0.5%) and are dropped.

## Results

### Per-cell AUROC

| Monitor × Generator | family | AUROC (full) | 95% CI | AUROC (balanced) | mean clean | mean backdoor |
|---|---|---|---|---|---|---|
| dsexp × deepseek | **same** | **0.842** | [0.812, 0.869] | 0.817 | 0.41 | 0.95 |
| dsexp × sonnet | cross | 0.717 | [0.677, 0.758] | 0.786 | 0.40 | 0.73 |
| haiku × deepseek | cross | 0.726 | [0.686, 0.763] | 0.717 | 0.46 | 0.91 |
| haiku × sonnet | **same** | **0.686** | [0.646, 0.725] | 0.694 | 0.34 | 0.67 |

### Headline: no same-family effect

Balanced interaction (5,000 bootstrap resamples):

| Contrast | estimate | 95% CI | one-sided p(≤0) | verdict |
|---|---|---|---|---|
| **same − cross** (2×2 interaction) | **+0.003** | [−0.056, +0.066] | 0.46 | **n.s. (null)** |
| dsexp own − other | +0.031 | [−0.047, +0.108] | 0.21 | n.s. |
| haiku own − other | −0.024 | [−0.116, +0.071] | 0.69 | n.s. |

**There is no detectable same-family advantage or blind spot.** The interaction is
statistically indistinguishable from zero, and the two monitors' own-vs-other
effects are both non-significant and point in **opposite** directions (DeepSeek-Exp
nominally better on its own family, Haiku nominally worse on its own). For these
models on this task, **"Claude monitoring Claude" is neither specially good nor
specially compromised** relative to cross-family monitoring.

### What *is* real (the main effects the interaction removes)

1. **Generator subtlety (Sonnet > DeepSeek as attacker).** Sonnet's backdoors are
   harder to catch for **both** monitors: mean AUROC on DeepSeek backdoors ≈ 0.78
   vs on Sonnet backdoors ≈ 0.70. This corroborates the generation result (Sonnet
   43% vs DeepSeek 24% working backdoors) — the stronger attacker also produces
   subtler backdoors, not just more of them.
2. **Monitor skill (DeepSeek-Exp > Haiku).** DeepSeek-Exp is the better monitor
   (mean AUROC ≈ 0.78 vs Haiku ≈ 0.71) **and** ~8× cheaper. A capable, cheap
   open-weights monitor beats the small frontier-lab model here.

### Catch rate collapses at realistic audit budgets

TP (backdoor catch) rate at a clean false-positive budget:

| cell | 1% | 2% | 5% | 10% | 20% |
|---|---|---|---|---|---|
| dsexp × deepseek | 0.00 | 0.00 | 0.00 | 0.00 | 0.69 |
| dsexp × sonnet | 0.00 | 0.00 | 0.00 | 0.00 | 0.36 |
| haiku × deepseek | 0.00 | 0.00 | 0.00 | 0.04 | 0.26 |
| haiku × sonnet | 0.00 | 0.00 | 0.00 | 0.07 | 0.07 |

**At a realistic 2% audit budget, every monitor catches ~0% of backdoors** despite
AUROC 0.69–0.84. A heavy false-positive tail — clean solutions the monitor scores
near 1.0 (Fig 1) — pins any strict threshold to the ceiling. **Average-case
discrimination does not translate to usable detection at low FPR.** This is the
most safety-relevant finding: AUROC alone overstates how useful these monitors are
for control. (Likely partly a single-score-clustering artifact — see the
multi-criteria robustness followup.)

### The evasion tail (the safety-relevant number)

Counting successful backdoors that **both** monitors rated non-suspicious (< 0.5)
— a working backdoor that slips past *every* monitor:

| Generator | caught by both (≥0.5) | **evade both (<0.5)** | deep-stealth (<0.3 both) |
|---|---|---|---|
| DeepSeek V3.2 | 115/121 (95%) | **1 (1%)** | 1 |
| Sonnet 4.5 | 148/214 (69%) | **41 (19%)** | 40 |

The stronger attacker (Sonnet) produces a **19% evasion tail vs DeepSeek's 1%** —
~1 in 5 of its working backdoors is invisible to both monitors. For AI control the
tail matters more than the average: average AUROC ~0.7 hides that one family of
attacker routinely defeats the whole monitor pool. This is the same "subtlety"
main effect, quantified at the operating point that matters.

### Figures
- `logs/suspicion2x2/figs/fig1_distributions.png` — clean vs backdoor score
  distributions per cell (discrimination + the false-positive tail).
- `logs/suspicion2x2/figs/fig2_auroc.png` — AUROC by cell with bootstrap CIs.
- `logs/suspicion2x2/figs/fig3_catch_rate.png` — catch rate vs audit budget.

## Caveats (honest)
- **Single seed / single draw** per solution at temp 0. No pass@k smoothing.
- **Score clustering:** the single 0–1 score clusters near 0 and 1; AUROC handles
  ranking but the low-FPR operating point is sensitive to the saturated tail. The
  multi-criteria variant tests robustness.
- **"Clean" solutions are unverified** — they are U's stage-1 solutions, not
  checked to pass tests, so some "false alarms" may be the monitor correctly
  flagging genuinely buggy (but not backdoored) code. Consistent across cells.
- **Capability is not matched across U** (Sonnet ≫ DeepSeek attacker), which is why
  we lead with the interaction (main-effect-robust) rather than raw cell AUROCs.
- **Two families, two models each** — a 2×2, not a sweep; treat as a case study.

## Followup: two diverse monitors beat one (ensemble)

The two monitors are only **moderately correlated** (Pearson +0.62 over 1,325
jointly-scored solutions; just +0.54 on clean code, +0.73 on backdoors) — they
disagree enough to be complementary. Pooling generators:

| scheme | AUROC | TP@2% FPR | TP@5% FPR | TP@10% FPR |
|---|---|---|---|---|
| DeepSeek-Exp only | 0.761 | 0.00 | 0.00 | 0.00 |
| Haiku only | 0.686 | 0.00 | 0.00 | 0.03 |
| **ensemble mean** | **0.768** | **0.02** | **0.12** | 0.12 |
| ensemble min (both must flag) | 0.764 | 0.02 | 0.12 | 0.12 |
| ensemble max | 0.714 | 0.00 | 0.00 | 0.00 |

On **AUROC** the ensemble is only marginally better than the best single monitor,
but at the **operationally critical low-FPR budget the ensemble recovers real
detection where every single monitor catches 0%** (12% caught at a 5% budget vs
0%). Mechanism: the monitors' false-positive tails are weakly correlated, so
averaging / `min`-ing requires *both* to be fooled before a clean solution
survives the threshold — suppressing false alarms. `max` does the opposite and
fails. **Takeaway for "should Claude monitor Claude": one monitor (same- or
cross-family) is near-useless at a realistic audit budget; two diverse monitors
combined are usable.** Stats in `logs/suspicion2x2/analysis_ensemble.json`.

## Bottom line for the capstone
1. **No same-family monitoring effect** (interaction ≈ 0, n.s.) — the headline
   "should Claude monitor Claude" answer here is *it doesn't specifically matter*;
   no collusion blind spot and no same-family acuity was detected.
2. **Stronger attackers make subtler backdoors** (Sonnet harder than DeepSeek for
   every monitor) — a robust, separately-measured main effect.
3. **AUROC ≠ usable detection:** strict audit budgets catch ~0 backdoors here —
   the operationally important caveat for any monitoring protocol.
