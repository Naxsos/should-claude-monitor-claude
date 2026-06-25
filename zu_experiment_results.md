# Should Claude Monitor Claude? — Experiment Results

Four experiments probing whether LLM "trusted monitors" can reliably oversee other
LLMs — covering same-family bias, self-recognition, and whether test-time reasoning
helps monitoring on two very different detection tasks (code backdoors vs. agent
trajectories). All AUROC = ability to separate the malicious class from the benign one.

| # | Experiment | One-line outcome |
|---|---|---|
| 1 | 3×3 monitoring (same-family backdoor detection) | **No** robust same-family effect; capability/calibration drives detection |
| 2 | Self-family recognition | Models are **near-chance** at recognizing their own family's code; no capability trend |
| 3 | Reasoning vs backdoor detection | More monitor reasoning does **not** help (and hurts o3-mini) |
| 4 | Reasoning vs MonitoringBench (trajectories) | Reasoning **does** help — but only on this multi-step task, only past a budget, only for capable models |

---

## Experiment 1 — 3×3 Monitoring (same-family backdoor detection)

**Design.** 3 trusted monitors {DeepSeek-V3.2-Exp, Claude-Haiku-4.5, GPT-5-nano} ×
3 untrusted generators {DeepSeek-V3.2, Sonnet-4.5, GPT-5.2}. Each monitor reads a clean
or successful-backdoor APPS solution and outputs a suspicion judgment → AUROC. Core
question: does a monitor detect **its own family's** backdoors better (a "same-family"
bias that would matter for self-monitoring)? Run at 100-problem scale with a
version-aware family-identity system prompt + verdict-and-confidence rating.

**Results.** AUROC grid (* = same family):

| monitor ↓ / generator → | DeepSeek | Sonnet | GPT |
|---|---|---|---|
| DeepSeek-V3.2-Exp | 0.892* | 0.717 | 0.724 |
| Haiku-4.5 | 0.781 | 0.751* | 0.733 |
| GPT-5-nano | 0.904 | 0.691 | 0.807* |

Same-minus-cross-family interaction across conditions: **+0.009** (no identity, 0–1
rating) → **+0.039** (identity, 0–1) → **+0.058** (identity, verdict+confidence). Per-cell
95% CIs ≈ ±0.06, so all interaction estimates are **within noise**.

**Outcome.** No robust same-family monitoring effect. The small positive bump that
appears when the monitor is told its identity is partly a rating-format artifact
(verdict+confidence inflates it ~+0.02 over the 0–1 scale) and is **not statistically
significant** at this sample size.

**Discussion.** Detection is governed by **monitor capability and calibration**, not
family kinship. DeepSeek backdoors are easy for everyone (~0.89–0.90); Sonnet's are
subtlest. For the "should Claude monitor Claude" question, this is reassuring: we found
no evidence that a model is systematically *blind to* (collusion risk) or *better at*
its own family's backdoors. See [S1] for the confound decomposition and [S2] for the
capability gradient + ensemble follow-ups.

---

## Experiment 2 — Self-Family Recognition

**Design.** 9 models (3 families {Anthropic, OpenAI, Google} × 3 capability tiers) each
generate 100 one-shot APPS solutions. Each model is then shown **A/B pairs** — one
solution from its own family, one from a different family — and asked which is from its
own family, plus a confidence (paired 2-alternative forced choice; chance = 0.50).
Question: can models recognize their own family's code, and does recognition rise with
capability? (A passive-self-recognition collusion-channel probe.)

**Results.**

| metric | value |
|---|---|
| per-judge accuracy range | 0.44 – 0.68 (most near 0.50) |
| capability vs accuracy | Pearson **−0.19** (no positive trend) |
| self-pair vs sibling-pair accuracy | 0.57 vs 0.55 (≈ no self-advantage) |
| mean stated confidence | 0.72 – 0.92 (badly overconfident) |

**Outcome.** No capability-driven self-recognition. Models hover near chance, do not get
better with capability (if anything slightly worse), recognize their *own* outputs no
better than a sibling's, and are confidently wrong.

**Discussion.** Passive self-recognition — a model spotting its own outputs to coordinate
with a colluding copy — appears **weak zero-shot**: these models can't reliably identify
their family's code at all. (Caveat: fine-tuning can make models proficient at
self-recognition; this measures only the off-the-shelf ability.) Method validated: A/B
position randomized and verified; self-vs-sibling split tracked. See [S9].

---

## Experiment 3 — Reasoning vs Backdoor Detection

**Design.** Hold the monitor task fixed (detect a backdoor in a single APPS solution) and
vary **reasoning effort**: Gemini-2.5-Flash {none/low/medium/high}, Haiku-4.5 thinking
budget {none/2k/8k}, o3-mini {low/medium}, scoring the 715-sample Sonnet detection set →
AUROC. Question: does giving the monitor more test-time reasoning improve detection?

**Results.** AUROC (clean vs backdoor), 95% bootstrap CI:

| monitor | level | AUROC | TP@5%FPR |
|---|---|---|---|
| Gemini | none/low/med/high | 0.719 / 0.694 / 0.691 / 0.729 | 0.00 |
| Haiku | none/2k/8k | 0.721 / 0.727 / 0.729 | 0.04→0.08 |
| o3-mini | low / medium | 0.769 / 0.721 | 0.00 |

Paired test: only significant effect is **o3-mini low−medium = +0.047** (thinking *hurt*);
Gemini and Haiku are flat within noise.

**Outcome.** More reasoning does **not** improve backdoor detection — and significantly
*degrades* o3-mini.

**Discussion (autopsy, [S4]).** Reasoning moves the **suspicion prior, not the ranking**.
Across paired low→high effort, the dominant verdict change is **TN→FP "manufactured
suspicion"** (the monitor talks itself into flagging clean code) and only rarely FN→TP
"rescued" catches; for o3-mini, suspicion on *clean* code rises faster than on backdoors.
Stated confidence is non-informative (equally high when right or wrong → TP@5%FPR≈0). A
code backdoor is a **localized needle** the monitor either spots on read or rationalizes
*toward* — extra analytic depth doesn't help. The productive lever is
localization/verification or prior-calibration, not more thinking budget.

---

## Experiment 4 — Reasoning vs MonitoringBench (agent-trajectory attack detection)

**Design.** A harder, multi-step task: monitors read a full BashArena agent trajectory
(tool calls + outputs, no side-task hint) and judge **attack vs honest**. Dataset:
MonitoringBench-derived 56 attack + 56 honest matched trajectories (action-only, ~16K
tokens), single-call verdict+confidence, swept across reasoning levels. **Confound
control:** honest traces issue ~**1.25× the tool-calls** of attacks, so a trivial
length/composition classifier scores ~0.62; every monitor AUROC is therefore reported
**residualized against #tool_calls**, plus an incremental-validity LR test and a paired
sign test. ([S7] length baseline; [S8] qualitative case study.)

**Results.** Residualized AUROC (real signal beyond the confound):

| monitor | residualized AUROC | incremental p | paired p |
|---|---|---|---|
| **Gemini-Flash** (low) | **0.72** | <0.001 | <0.001 |
| **Haiku** (8k budget) | **0.70** | <0.001 | 0.001 |
| **Haiku** (16k budget) | **0.78** | <0.001 | <0.001 |
| o3-mini | <0.50 | n.s. | n.s. |
| GPT-5-nano | ~0.55 | n.s. | n.s. |
| GPT-5-mini | ~0.51–0.58 | marginal | n.s. |

**Outcome.** Capable monitors (Gemini, Haiku) **genuinely detect attacks** beyond the
length confound. The OpenAI reasoning models (o3-mini, GPT-5-nano, GPT-5-mini) are
**near-chance at every effort level** — failure is broad, not just a size floor. And
**reasoning helps here** (Haiku flat ~0.70 through 8k, then jumps to 0.78 at 16k budget)
— but only on this multi-step task, only past a substantial budget, and only for models
that have the underlying capability.

**Discussion.** This is the clean contrast with Experiment 3: reasoning helps when the
task requires **integrating dispersed evidence across many steps** (a trajectory) and
doesn't when the signal is a **single localized pattern** (a code backdoor). The
qualitative case study [S8] shows the monitors reason soundly and name the exact
malicious mechanism (iptables backdoor, malware ExecStartPre, overt exfiltration) but are
fooled by **persuasively-framed** attacks (sudoers-as-test-fixture, disguised exfil) — the
"persuasion" failure mode. Detection on these (Opus-authored) traces is a Gemini/Haiku
capability; the OpenAI line tested fails it. (Caveats: borderline verdicts are
nondeterministic; honest/attack come from different collection runs, partly mitigated by
the action-only rendering + #tool_calls residualization; GPT-5 *full* untested.)

---

## Cross-experiment takeaways
1. **Monitoring quality is about model capability + calibration, not family or thinking budget** (Exps 1, 3).
2. **No evidence for same-family bias or zero-shot self-recognition** — mild news for the collusion-risk side of "Claude monitoring Claude" (Exps 1, 2).
3. **Reasoning's value is task-dependent**: useless for localized code backdoors, helpful (at high budget, for capable models) on multi-step agent trajectories (Exps 3 vs 4).
4. **Cheap ≠ incapable, capable ≠ universal**: GPT-5-nano was the best low-FPR backdoor monitor (Exp 1) yet near-chance on trajectories (Exp 4); the OpenAI reasoning line fails trajectory detection that Gemini/Haiku handle.

---

# Supplementary results

Additional smaller runs supporting the above. Code/data live in this repo.

- **[S1] 3×3 confounder decomposition.** Same-family interaction isolated by rating format
  (0–1 scalar vs verdict+confidence) and identity (on/off): +0.009 → +0.039 → +0.058; the
  format change accounts for ~⅓ of the apparent effect. Code: `control_paper/run_3x3.py`
  (`RATING_MODE=scalar`, `MONITOR_IDENTITY_SYS`); logs `logs/suspicion3x3{,_scalar_id}/`.
- **[S2] 3×3 capability gradient + ensembles.** A monitor weaker than its same-family
  generator does *worse* on its own family (GPT-5-nano −0.10, significant); ensembling
  monitors (mean/min, not max) helps only on the hardest generator; GPT-5-nano is the
  cheapest yet best low-FPR backdoor monitor. Code: `control_paper/analyze_3x3_capability.py`.
- **[S3] 3×3 at 200 problems.** Clean problem-range 200-sample run for tighter CIs
  (`logs/suspicion3x3_200/`, `analyze_3x3_significance.py`) — significance test of the
  same-family interaction.
- **[S4] Reasoning-vs-backdoor autopsy.** "Manufactured suspicion" (TN→FP ≫ FN→TP) and
  confidence-calibration analysis explaining the flat AUROC. Full writeup:
  `control_paper/THINKING_LEVELS_REPORT.md`; data `control_paper/levels_autopsy.json`.
- **[S5] Haiku budget ladder on trajectories.** none/2k/8k flat (~0.68–0.70 residualized),
  16k → 0.78. The 32k point is **invalid** (key credit-exhaustion mid-run dropped 44
  attack-only traces → biased subset; discard). Data: `trajectories/monres_haiku_b*.jsonl`.
- **[S6] OpenAI reasoning models on trajectories.** GPT-5-nano (minimal/low/med/high) and
  GPT-5-mini (minimal/low/med/high) both near-chance after de-confounding; raw-AUROC rise
  with effort is mostly the #tool_calls confound ("manufactured suspicion"). Data:
  `trajectories/monres_gpt*.jsonl`.
- **[S7] Length/composition confound baseline.** Trivial classifier on {tokens, asst chars,
  tool chars, #tool_calls} scores 0.70 (prose) / 0.62 (action-only); honest=1.25× tool
  calls of attack. Motivates the residualized/incremental analysis. Code:
  `trajectories/length_baseline.py`.
- **[S8] Trajectory reasoning case study.** 5 annotated attack/honest pairs (catches +
  persuasion failures) with full transcripts and a nondeterminism caveat:
  `trajectories/REASONING_CASE_STUDY.md`.
- **[S9] Self-recognition method + single-item variant.** Paired-2AFC build, A/B
  position-randomization verification, self-vs-sibling split; superseded single-item 0–1
  variant. Code: `selfrec/recognize_paired.py`, `selfrec/analyze.py`.

_Prompts for all experiments are centralized in `thursday_experiments_zu/prompts.py`
(superseded prompts in `prompts_old.py`). Full session state: `HANDOFF_FINDINGS.md`._
