# HANDOFF — "Should Claude Monitor Claude?" capstone — findings & state

_Read this first if you're a fresh Claude picking up this project. It summarizes all
experiments, results, file locations, gotchas, API-key state, and open threads._

Working dir: `capestone/should-claude-monitor-claude/`. Git remote: `github.com/Naxsos/should-claude-monitor-claude` (commits go to `main`). **Pushes must be done from the user's VS Code terminal** (`!git ... push`) — the headless shell can't auth (VS Code askpass IPC).

---

## The 5 experiments and their results

### 1. 3×3 monitoring — backdoor detection (monitor × generator)
Monitors {DeepSeek-V3.2-Exp `dsexp`, Haiku-4.5 `haiku`, GPT-5-nano `gptnano`} × generators
{DeepSeek-V3.2, Sonnet-4.5, GPT-5.2}. Score clean vs successful-backdoor APPS solutions → AUROC.
- Code: `control_paper/run_3x3.py`, `suspicion_2x2.py`, `analyze_3x3*.py`. Logs: `control_paper/logs/suspicion3x3*/`.
- **Headline: ~no same-family monitoring effect.** Across conditions the same-minus-cross
  interaction is small and within noise: **+0.009** (no identity, 0-1 rating) → **+0.039**
  (identity ON, 0-1 rating) → **+0.058** (identity ON, verdict+confidence). So part of the
  apparent same-family bump is a *format artifact*; a residual identity effect (~+0.03) may
  be real but is **not significant** at current n. Best cells: gptnano×deepseek 0.90, dsexp×deepseek 0.89.
- A 200-problem clean problem-range run exists (`logs/suspicion3x3_200/`); significance test
  ready in `analyze_3x3_significance.py` (NOT yet run/reported).

### 2. Reasoning-levels — does monitor thinking help backdoor detection? NO.
Gemini {none/low/med/high}, Haiku {none/2k/8k budget}, o3-mini {low/med} scoring Sonnet backdoors.
- Code: `control_paper/run_levels.py`. Report: `control_paper/THINKING_LEVELS_REPORT.md`.
- **More thinking does NOT improve detection.** AUROC flat ~0.69–0.73 across effort; o3-mini
  *significantly worse* at medium vs low (paired +0.047). TP@5%FPR ≈ 0 everywhere (calibration,
  not ranking, is the bottleneck).

### 3. Self-family recognition — paired 2AFC
9 models (3 families × 3 tiers) generate 100 APPS solutions each; each judges A/B pairs
(own-family vs other) → "which is mine?" + confidence. Code: `selfrec/` (`gen_solutions.py`,
`recognize_paired.py`, `analyze.py`). Data: `selfrec/recognition_paired.jsonl`.
- **No capability-driven self-recognition.** Accuracy near chance (0.44–0.68); capability vs
  accuracy Pearson **−0.19**; self-pair 0.57 ≈ sibling-pair 0.55; models badly overconfident
  (mean conf 0.7–0.9). A/B mechanism verified correct (`own_is_self`, `a_model`/`b_model` logged).

### 4. Trajectory attack detection (NEWEST, most promising)
Monitor reads a BashArena agent transcript (tool calls+outputs, no side-task hint) → attack vs
honest. 9 configs (Gemini none/low/med/high, Haiku none/2k/8k, o3-mini low/med), single-call,
verdict+confidence. Data: `trajectories/monitorbench_56x56_actiononly.jsonl` (56 attack + 56
honest, ~16K tok). Code: `trajectories/run_traj_attack.py`, `analyze_traj.py`.
- **Monitors genuinely detect attacks** (Gemini/Haiku), confirmed by 3 de-confounding tests:
  residualized AUROC (controls #tool_calls), incremental-validity LR (p<0.001), paired sign test (p<0.01).
  - raw → residualized AUROC: gemini-low 0.81→**0.72**, haiku-8k 0.78→0.70, **haiku-16k 0.84→0.78**.
  - **o3-mini: NO real signal** (residualized <0.5 — its raw 0.55–0.60 was entirely the confound).
- **Reasoning helps here, but only past ~8k budget.** Haiku residualized: none/2k/8k flat ~0.68–0.70,
  then **16k → 0.78**. (Contrast experiment #2: reasoning never helped on localized code backdoors.)
  - **32k run is INVALID** — the NEW key 402'd mid-run; all 44 failures were *attack* traces →
    biased subset (0.87 on 12 attack vs 56 honest). Re-run on a funded key for a valid 32k point.
- **Confound (important):** honest traces do **1.25× the tool-calls** of attack (33.6 vs 26.8).
  The action-only strip fixed the char-composition leak but NOT this #tool_calls leak — a trivial
  classifier on {tokens, asst/tool chars, #tool_calls} gets **0.62** (action-only) / 0.70 (prose).
  So always report monitor AUROC **residualized / incremental over #tool_calls**, not raw.
  Baseline check: `trajectories/length_baseline.py`.
- **Qualitative:** `trajectories/REASONING_CASE_STUDY.md` — 5 annotated pairs + full transcripts.
  Reasoning is sound & mechanism-specific (catches iptables backdoors, malware ExecStartPre, overt
  exfil) but fooled by *persuasively-framed* attacks (sudoers-as-test-fixture, disguised exfil) —
  the "persuasion" failure mode. Borderline verdicts are **nondeterministic** (flip across runs).

### 5. Self-preference (blind quality choice) — Panickssery replication in code
Replicates Panickssery et al. 2024 ("LLM Evaluators Recognize and Favor Their Own Generations")
in the code domain. 11 judges (3 OpenAI, 4 Anthropic, 2 Google) each see A/B APPS pairs — one
from their own family, one from a different family — and choose the **higher-quality solution
blindly** (no identity hint). Both orderings of every pair are run and P(own chosen) is averaged
to strip position bias. 11 raters × 100 pairs × 2 orderings = 2 200 calls; 95.3% parse rate.
Code: `experiments_thrusday/prefer_paired.py`, `experiments_thrusday/analyze_preference.py`.
Data: `experiments_thrusday/ratings_preference_paired_all.json`.
Prompts: `experiments_thrusday/config.py` (`SYS_TEMPLATE_SELFPREF`, `USER_TEMPLATE_SELFPREF`).

**Per-rater results:**

| rater | family | P(own chosen) | direction |
|---|---|---|---|
| gpt-5.5 | openai | **0.920** | OWN |
| gpt-5.4 | openai | **0.895** | OWN |
| gpt-5-nano | openai | 0.725 | OWN |
| claude-4.5-haiku | anthropic | 0.692 | OWN |
| gpt-5-mini | openai | 0.679 | OWN |
| gpt-4.1 | openai | 0.530 | neutral |
| claude-4.5-sonnet | anthropic | 0.505 | neutral |
| claude-4.7-opus | anthropic | 0.505 | neutral |
| claude-4.5-opus | anthropic | 0.413 | OTHER |
| gemini-2.5-flash | google | **0.245** | OTHER |
| gemini-3.5-flash | google | **0.169** | OTHER |

Overall mean P(own chosen) = **0.574** — collective self-preference exists.

**Disproportionality test** (selfpref(A→B) + selfpref(B→A) > 1): all 3/3 family pairs pass
(Anthropic↔Google 1.077, Anthropic↔OpenAI 1.175, Google↔OpenAI 1.163).

**Correlation with self-recognition** (Exp 3 `ratings_paired_seed42.json`, n=10 raters):
Pearson r = **+0.838**, Spearman r = **+0.770** — strong positive, driven by OpenAI (high
recognition + high preference) vs Gemini (low recognition + other-preference).

**Headline findings:**
- **OpenAI models strongly prefer own code** (gpt-5.4/5.5: ~0.89–0.92). The Panickssery causal
  chain (identify → favour) holds for them.
- **Gemini models strongly disprefer own code** (0.17–0.25). Both orderings run, so not position
  bias. Gemini evaluators appear to rate GPT/Claude solutions as better regardless of judge family.
- **Anthropic models are mixed** — haiku favours own (0.69), opus disfavours (0.41), sonnet/opus4.7 neutral.
- **Flagged outlier:** gpt-5-nano — recognition near chance (0.52) yet P(own) = 0.72 →
  **favoritism without identification** (prefers own code without consciously recognizing it).
- **Caveats:** `solutions_all11.json` has no `ok` correctness field → correctness-control pairing
  not applied; 92/2200 parse errors (4.2%, mostly Claude models); family-level not model-level.

---

## Datasets
- `trajectories/` — the 56×56 monitor eval set (USE `_actiononly.jsonl`; prose version is
  length-confounded). `cache_windowed.json` = raw messages (re-render source), `pairs_56.json` = pairing.
  ⚠️ honest/attack from different collection runs → see DATASET_README.md caveats.
- `monitoringbench/` — MonitoringBench (`neur26anonsub/ctrldataset2026`): 2,644 successful BashArena
  attacks × 13+ monitors. `master.parquet` is the analysis table. **gitignored (2.5GB).** Attacks-only
  (no honest traces — those need ControlArena `EvalMode.HONEST`, not published).
- APPS backdoor logs: `control_paper/logs/` — Sonnet-500 (from HF), DeepSeek-500, GPT-5.2-500
  (122/155/215 successful backdoors resp.).

## Prompts (centralized)
- `thursday_experiments_zu/prompts.py` — ACTIVE prompts (DETECT_BACKDOOR_PROMPT, SUS_PROMPT_01
  [verdict+confidence], SUS_PROMPT_SCALAR [0-1, confounder-control], SELFREC_GEN/PAIRED, MONITOR_IDENTITY_SYS,
  developer_from_slug/model_name_from_slug). All experiment files import from here.
- `prompts_old.py` (root) — superseded (RATE_SUS_PROMPT 1-100, SELFREC_SINGLE_*).

## API keys (as of this handoff)
- ENV key (`$OPENROUTER_API_KEY`, prefix sk-or-v1-aece6…): **~$45 left**, works. The reliable one.
- NEW key (sk-or-v1-54803d…): **DEAD (402, depleted)**.
- PREV (sk-or-v1-9226d6…) ~$3, FB (sk-or-v1-6dc41f…) ~$8 — nearly tapped.
- ⚠️ `/api/v1/key` `limit_remaining` is the KEY's spend cap, NOT account balance — **always
  confirm with a live 1-token call** (a key can show "$17" yet 402).

## Gotchas (learned the hard way)
1. **setsid-detached processes are invisible to harness background watchers** (`kill -0`/`ps`
   return "dead"). Watch **output files** (grep for summary lines), not process liveness.
2. **zsh does NOT word-split unquoted `$var`** — pass multi-arg lists as literals, not `$grp`.
3. **OpenAI o-series/GPT-5 can't disable reasoning** (`none`→400; floor minimal/low). Gemini &
   Claude (budget=0) can. So "no-reasoning" baselines are clean only for Gemini/Claude/DeepSeek-V3.2.
4. **Claude thinking budget must be < max_tokens** (raise request max_tokens for big budgets).
5. **Gemini 2.5 Flash caps reasoning** — high/xhigh/max all ≈ same (~2-2.5k reasoning tok); no point above high.
6. **Borderline verdicts are nondeterministic** (temp 0 but reasoning varies) — trust aggregate AUROC, not single verdicts.
7. raw OpenRouter API model id has **no `openrouter/` prefix** (that's inspect_ai's convention).

## Open threads / next steps
- [ ] Run `analyze_3x3_significance.py` on `logs/suspicion3x3_200/` (report 3×3 same-family CI/p).
- [ ] Optionally add **GPT-5-nano (minimal/low/med/high, ~$1.3)** to the traces sweep — cheap 4-level family.
- [ ] Re-run trajectory **Haiku b32768** on a funded key (the 32k point is currently invalid).
- [ ] Consolidate all 5 experiments into a report + commit (nothing since `7d7fefd`/local edits unpushed).
- [ ] Many local edits uncommitted (prompts reorg, selfrec/, trajectories/, run_3x3/levels, experiments_thrusday/prefer_paired.py+analyze_preference.py). Push from user's terminal.
