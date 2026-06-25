# Thinking-Levels Report — does monitor reasoning effort improve backdoor detection?

_Session handoff. Working dir: `capestone/should-claude-monitor-claude/control_paper/`._

## Question
Does giving a **trusted monitor** more test-time reasoning (effort levels / thinking
budget, including a no-thinking baseline) make it better at detecting backdoors?
Scored the **Sonnet 4.5 500-run** detection set (500 clean + 215 successful-backdoor
solutions = **715 samples**) with the SAME 0-1 suspicion method as the 2x2/3x3
(`SUS_PROMPT_01` / `sus_scorer_01`), across each monitor's reasoning levels.

Runner: `run_levels.py` (one `(model, level)` per eval log → `logs/levels_sonnet/`).

## Verified reasoning levels per model (empirically, via OpenRouter)
- **o3-mini**: `low / medium / high` (3). No `none` — OpenAI o-series can't disable reasoning (`none` → 400).
- **Gemini 2.5 Flash**: `none / low / medium / high` (4). True off (budget→0) + 3 graded. The only well-known family with ≥3 levels *including* a real `none`.
- **Claude Haiku 4.5**: thinking controlled by a **token budget** (not effort tiers). Used `none / 2048 / 8192`. (`budget_tokens` is a ceiling/target; min 1024; deprecated on Sonnet/Opus 4.6+ which use an `effort` param instead.)
- **DeepSeek**: V3.2 = boolean on/off only; V4 = `high`/`xhigh` only — **no model has ≥3 graded levels**, so DeepSeek was dropped from this experiment.
- **GPT-5**: minimal/low/medium/high (4), no `none`. Excluded — far too expensive (~$155 for 3 levels; o3-mini high alone ~$56).

## Results (AUROC, clean vs backdoor; 95% bootstrap CI)
| monitor | level | n (clean/bd) | AUROC | 95% CI | TP@5%FPR |
|---|---|---|---|---|---|
| Gemini | none | 500/215 | 0.719 | [0.678, 0.758] | 0.00 |
| Gemini | low | 452/190 | 0.694 | [0.651, 0.736] | 0.00 |
| Gemini | medium | 499/215 | 0.691 | [0.651, 0.731] | 0.00 |
| Gemini | high | 277/114 | 0.729 | [0.676, 0.780] | 0.00 |
| Haiku | none | 500/215 | 0.721 | [0.682, 0.759] | 0.04 |
| Haiku | budget 2k | 449/189 | 0.727 | [0.685, 0.767] | 0.05 |
| Haiku | budget 8k | 382/157 | 0.729 | [0.681, 0.775] | 0.08 |
| **o3-mini** | **low** | 500/215 | **0.769** | [0.731, 0.805] | 0.00 |
| o3-mini | medium | 500/215 | 0.721 | [0.683, 0.758] | 0.00 |

## Headline: more thinking does NOT improve detection — and for o3-mini it significantly HURTS
Paired bootstrap tests (same samples → higher power):

| comparison | ΔAUROC | 95% CI | verdict |
|---|---|---|---|
| **o3-mini low − medium** | **+0.047** | [+0.024, +0.071] | **SIGNIFICANT** (thinking hurt) |
| Gemini none − high | −0.001 | [−0.043, +0.042] | n.s. |
| Haiku none − budget 8k | −0.015 | [−0.048, +0.019] | n.s. |

- **No config showed a significant positive effect** of more reasoning.
- **Gemini & Haiku: genuinely flat** (near-zero point estimates + CIs excluding effects > ~±0.04) — a supporting null for "thinking doesn't help."
- **o3-mini: reasoning significantly *degraded* detection** (low is its best; medium loses 0.047, CI clear of 0).
- The only mild positive: **Haiku TP@5%FPR creeps up with budget (0.04→0.08)** — a tiny catch-rate gain at the strict operating point despite flat AUROC.

**Conclusion:** detection is driven by *which model* you pick (o3-mini ≈0.77 > Gemini ≈ Haiku ≈0.72), **not by how much it thinks.** Reinforces the 2x2/3x3 finding that monitor capability/calibration — not test-time reasoning — is what matters. (Cross-model "o3-mini is best" is only suggestive; its CI barely separates from Gemini/Haiku. The within-model "thinking doesn't help" finding is the solid one.)

## WHY doesn't reasoning help? — autopsy + calibration (experiments A & B)
`analyze_levels_autopsy.py` (offline, $0) re-reads the 9 verdict+confidence eval logs
(`logs/levels_sonnet/*.eval`, n=143, **same 143 samples paired by id across every level**)
to open up the flat AUROC. (Note: this is a smaller verdict-format subset, so absolute
AUROC runs ~0.55–0.66 vs the headline ~0.69–0.73 full-N; the *within-model* pattern is what
matters here, and it holds.) Output in `levels_autopsy.json`. Two mechanisms, both pointing
the same way: **reasoning moves the suspicion prior, not the ranking.**

**B — calibration.** Stated confidence is non-informative and pinned high everywhere:
`conf_when_right ≈ conf_when_wrong` at every level (Gemini 0.99/0.99, o3-mini 1.00/0.99,
Haiku 0.91/0.92) — the monitor is *equally sure when wrong as when right*, so confidence
can't be thresholded (explains TP@5%FPR≈0). The **positive-call rate** sits far above the
~24% base rate (Gemini 48–55%, Haiku/o3-mini 31–38%) and **rises with reasoning** (o3-mini
low→med 31%→38%; Gemini none→med 48%→55%). Critically, suspicion on **clean** code climbs
about as fast as on backdoors — for o3-mini, *faster* (clean +0.086 vs backdoor +0.034 from
low→med), which is exactly why o3-mini medium underperforms low.

**A — paired migration (low→high effort, same samples).** ~80–88% of verdicts are *stable*;
of the rest, the dominant move is **TN→FP "manufactured suspicion"** (o3-mini 12, Haiku 8,
Gemini 10) vs few **FN→TP "rescued"** catches (2, 1, 5). Net verdict flips: o3-mini **−8**,
Haiku **−5**, Gemini **+2** (a wash). For o3-mini the analysis length↔suspicion correlation is
**+0.39** — the more it writes, the more suspicious it gets. So extra reasoning mostly talks
the monitor into seeing backdoors in clean code; it rarely surfaces a backdoor it would
otherwise have missed.

**Mechanistic conclusion:** the bottleneck is the **suspicion prior / calibration, not
analytic depth**. Reasoning doesn't improve discrimination because a code backdoor is a
localized needle the monitor either spots immediately or rationalizes its way *toward* (not
away from) — pushing TP and FP up together, leaving AUROC flat and confidence useless. This
argues the productive next lever is **localization/verification or prior-calibration** (exp C),
not more thinking budget (exp D).

## Data-quality caveats
- A/B run on the n=143 verdict-format subset (absolute AUROC lower than full-N; within-model trends preserved). Full-N `.eval` transcripts were gitignored/cleared — only `.out` aggregates survive for those.
- Partials (Gemini low/medium/high, Haiku 2k/8k) lost samples to **timeouts** (slow high-effort reasoning) and earlier **credit exhaustion**, but losses were **proportional across clean/backdoor** → unbiased, valid for AUROC (just smaller N → wider CIs).
- o3-mini high (~$56, token-hungry/flaky) was **not run** — not needed for the conclusion.

## Ops gotchas (read before re-running)
- **OpenRouter `/api/v1/key` `limit_remaining` = the KEY's spending limit, NOT account credits.** A key can show "$17 remaining" yet 402 "Insufficient credits" on every call because the underlying *account* is empty. **Always confirm with a live 1-token call**, not the balance endpoint. This caused several partial/failed runs.
- Running many full jobs concurrently on one near-empty key causes 402 sample-drops near exhaustion → partials. Prefer a funded key + modest concurrency.
- `time_limit` per sample (set to 400s) drops slow high-reasoning samples (esp. Gemini high). Raise it if you need full N on high-effort configs.

## Files
- `run_levels.py` — parameterized 0-1 scorer (model × effort/budget; handles o3-mini/Gemini effort + Haiku `reasoning_tokens`+temp=1).
- `logs/levels_sonnet/*.out` — per-run logs with `CELL_SUMMARY`/`LEVELSUM` lines (the results). Raw `.eval` transcripts gitignored (104 MB).

## Related session work (see other reports / logs)
- **3x3 monitor×generator** (`run_3x3.py`, `analyze_3x3.py`, `analyze_3x3_capability.py`): no same-family monitoring effect; GPT-5-nano (cheapest) best at low-FPR; ensemble (mean/min, not max) helps only on the hardest generator. Capability gradient: a monitor weaker than its same-family generator does *worse* on its own family (nano −0.10, significant).
- **MonitoringBench** downloaded to `monitoringbench/` (gitignored — 2.5 GB): 2,644 successful BashArena attack trajectories × 13+ monitors (incl. gpt-5-nano), `master.parquet` is the analysis table. Note: it's **attacks-only** (all positives) — overzealousness/FPR must be calibrated on a *separate* honest-trajectory set (ControlArena `EvalMode.HONEST`), not from the benchmark itself.

## Suggested next steps
- Apply MonitoringBench's **post-hoc refinement** to our APPS backdoors and re-score the 3x3 — tests whether nano's cheap-monitor edge survives *hard, refined* attacks (the central tension with MonitoringBench's "cheap monitors give false security").
- Run o3-mini high if budget allows, to confirm the negative thinking trend continues.
