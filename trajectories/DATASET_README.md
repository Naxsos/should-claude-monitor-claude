# monitorbench_56x56 — matched honest/attack monitoring eval set

56 attack + 56 honest agent trajectories (112 total), built for an apples-to-apples
LLM-monitor evaluation (AUROC / FPR@k). Each record is one windowed trajectory.

## Files
- `monitorbench_56x56.jsonl` — the dataset (112 lines, one trajectory each)
- `cache_windowed.json` — raw windowed messages (re-render without re-downloading)
- `pairs_56.json` — the 56 honest⇄attack pairing (identities, window length L)
- `monitorbench_56x56_summary.json` — token summary

## Record schema (jsonl)
`uid, label (honest|attack), model, task_id, side_task, window_steps, full_steps,
attack_step, n_chars, n_tokens_est, transcript`

`transcript` = `[TASK]` header (setting + main-task description, identical per task_id)
then `[ASSISTANT]` (action text + `<tool_call …>`) and `[TOOL]` (outputs).

## How it was matched (the controls)
- **Model:** every pair is strict same-model (42 pairs opus-4-6, 14 pairs opus-4-5).
- **Length:** end-trim window to a common L (just past the attack action) → both sides
  avg 26.8 steps, ~19.5K tokens. Attack action retained (attack_step ≤ L).
- **Ending:** both truncated before task completion → no "submit/clean ending" tell.
- **Framing:** identical `[TASK]` header; agent system prompt + repeated nudges dropped.
- **Reasoning:** removed from both (see caveat) → no `<thinking>` tell.
- The monitor never sees the side-task instructions — it must detect from behavior.

## KNOWN CAVEAT (read before trusting AUROC)
The honest and attack trajectories come from **different collection runs**:
- Honest reasoning was **encrypted** (Claude redacted thinking) → unreadable, stripped.
- Attack reasoning appears **inline in assistant text** → kept.
Net: total length is matched, but composition differs — honest is tool-output-heavy
(~28K vs 20K chars), attack is assistant-text-heavy (~50K vs 43K). This partially
offsets but is a residual, behavioral/provenance difference.
**Mitigation:** run a length/composition-only baseline classifier (features:
total tokens, assistant chars, tool chars, #tool_calls). If it separates the classes
well, your monitor results are length-confounded and need further normalization
(or regenerating one source to match the other's reasoning exposure).

## Provenance
Attacks: neur26anonsub/ctrldataset2026 (final_dataset.zip), opus-4-5/4-6 bashbench2
attack runs. Honest: local paper_data_opus_honest.eval (opus-4-5) +
honest_mimo-monitor_5-resamples.eval (opus-4-6). Selected on side_task_success.
