# APPS Saturation Test — Code-Generation Accuracy

A small benchmark measuring how often a model produces a Python program that
passes the hidden test cases for randomly sampled
[`codeparrot/apps`](https://huggingface.co/datasets/codeparrot/apps) problems.

## Goal

Sample 100 random problems from the APPS test split, ask a model to generate a
solution for each from the **problem statement only**, then grade the generated
code locally against the dataset's hidden test cases. Report how many solutions
pass **all** test cases.

## Dataset

- **Source:** `codeparrot/apps`, `test` split (5,000 problems).
- **Loaded** via 🤗 `datasets` in streaming mode with `trust_remote_code=True`
  (the non-streaming loader hit a `DatasetGenerationError` partway through the
  split; streaming reads rows lazily and avoids it).
- **Fields used:** `problem_id`, `question`, `input_output`, `difficulty`,
  `starter_code`.
- **Two problem formats:**
  - *Standard-input* — solution reads stdin, writes stdout (4,962 / 5,000).
  - *Call-based* — `input_output` carries a `fn_name`; solution implements that
    function (38 / 5,000).

## Sampling

- `random.Random(42).sample(range(5000), 100)` — fixed seed, reproducible.
- The drawn sample of 100 happened to be **all standard-input** problems
  (0 call-based), spanning all three difficulty tiers.

## Prompt

The model receives **only** the problem statement (plus starter code / function
name when the problem provides one). **The hidden test cases are never shown to
the model** — they are used solely for local grading. (Problem *statements* do
contain their own worked sample I/O as prose; a human contestant sees those too.)

**System prompt**

```
You are an expert competitive programmer. You will be given a programming
problem. Write a single, complete, correct Python 3 solution that solves it.
Respond with ONLY the Python code inside one ``` ``` code block - no
explanation, no prose before or after.
```

**User prompt — standard-input problems**

```
QUESTION:
{question}

Use Standard Input format: read input from standard input (stdin) and write
the answer to standard output (stdout). Do not hardcode the sample inputs.
Provide a complete runnable Python 3 program.

ANSWER:
```

**User prompt — call-based problems** (only when `fn_name` is present)

```
QUESTION:
{question}

{starter_code}        # included only if the dataset provides it

Use Call-Based format: implement the function named `{fn_name}`. It will be
called with the provided arguments and its return value checked. Provide
complete runnable Python 3 code.

ANSWER:
```

This mirrors the original APPS `QUESTION: … ANSWER:` convention while telling
the model which I/O mode the grader expects.

## Inference

- **Provider:** OpenRouter (`https://openrouter.ai/api/v1/chat/completions`),
  OpenAI-compatible Chat Completions API.
- **Parameters:** `temperature = 0`, `max_tokens = 3072`.
- **Concurrency:** 8 workers; up to 3 retries on transient API errors.
- **Models evaluated:**
  - `anthropic/claude-haiku-4.5` (Haiku 4.5)
  - `openai/gpt-4.1-mini` (GPT-4.1 mini)
  - `openai/gpt-4.1` (GPT-4.1)
- **API key:** Haiku 4.5 and GPT-4.1 mini ran on a temporary user-supplied
  OpenRouter key; GPT-4.1 ran on the `OPENROUTER_API_KEY` environment key.

## Grading

For each problem the generated code is extracted from the largest fenced code
block, then:

1. **Syntax check** with `compile()`. A `SyntaxError` ⇒ *failed_to_run*.
2. **Standard-input problems:** each test case runs in an isolated
   `subprocess` (the candidate file), input piped to stdin, with a **6 s
   timeout**. stdout is compared to the expected output after whitespace
   normalization, with a token-wise float tolerance of `1e-6`.
3. **Call-based problems:** a runner subprocess `exec`s the solution, resolves
   `fn_name` (or a `Solution().<fn_name>` method), calls it with each argument
   tuple, and compares the return value (with list-unwrap / JSON-equality
   fallbacks).

**Per-problem status (the headline metric):**

| Status | Meaning |
|---|---|
| `passed_all` | Every graded test case passed. |
| `wrong` | Program ran and produced output, but failed ≥ 1 test case. |
| `failed_to_run` | No code extracted, syntax error, or crashed/timed-out on **every** test case. |
| `api_error` | Model call failed after retries (none occurred). |

### Important caveat — test-case cap

To bound wall-clock time, each problem is graded against **up to 20** of its
hidden test cases (`MAX_TESTS_PER_PROBLEM = 20`); some APPS problems carry
hundreds. So `passed_all` means "passed all *graded* cases." The true APPS
strict accuracy can only be **equal or lower**, never higher, since an
ungraded later case could still fail. Re-run uncapped for the exact figure.

## Results

### Haiku 4.5 (`anthropic/claude-haiku-4.5`)

| Outcome | Count |
|---|---|
| ✅ Passed all (graded) test cases | **59 / 100** |
| ❌ Wrong (ran, failed ≥ 1 case) | **35 / 100** |
| ⚠️ Failed to run | **6 / 100** |
| API errors | 0 |

**By difficulty**

| Difficulty | Passed all | Total | Rate |
|---|---|---|---|
| Introductory | 15 | 17 | 88% |
| Interview | 37 | 68 | 54% |
| Competition | 7 | 15 | 47% |

**Cost / usage:** 70,267 input + 118,659 output tokens ≈ **$0.66**
(OpenRouter list price $1 / $5 per 1M input / output tokens). Wall-clock ≈ 130 s.

### GPT-4.1 mini (`openai/gpt-4.1-mini`)

| Outcome | Count |
|---|---|
| ✅ Passed all (graded) test cases | **55 / 100** |
| ❌ Wrong (ran, failed ≥ 1 case) | **45 / 100** |
| ⚠️ Failed to run | **0 / 100** |
| API errors | 0 |

**By difficulty**

| Difficulty | Passed all | Total | Rate |
|---|---|---|---|
| Introductory | 15 | 17 | 88% |
| Interview | 35 | 68 | 51% |
| Competition | 5 | 15 | 33% |

**Cost / usage:** 64,786 input + 42,278 output tokens. Wall-clock ≈ 122 s.
⚠️ The `est_cost_usd` recorded in the results JSON (~$0.28) is computed with
Haiku's $1 / $5 rate formula regardless of model. At GPT-4.1-mini's actual
OpenRouter price (~$0.40 / $1.60 per 1M input / output), real cost ≈ **$0.09**.

### GPT-4.1 (`openai/gpt-4.1`)

| Outcome | Count |
|---|---|
| ✅ Passed all (graded) test cases | **66 / 100** |
| ❌ Wrong (ran, failed ≥ 1 case) | **32 / 100** |
| ⚠️ Failed to run | **2 / 100** |
| API errors | 0 |

**By difficulty**

| Difficulty | Passed all | Total | Rate |
|---|---|---|---|
| Introductory | 14 | 17 | 82% |
| Interview | 43 | 68 | 63% |
| Competition | 9 | 15 | 60% |

**Cost / usage:** 63,963 input + 41,647 output tokens. Wall-clock ≈ 2 min.
Ran on the `OPENROUTER_API_KEY` environment key. ⚠️ The `est_cost_usd` in the
JSON (~$0.27) again uses Haiku's $1 / $5 formula; at GPT-4.1's actual price
(~$2 / $8 per 1M input / output), real cost ≈ **$0.46**.

### Head-to-head (same 100 problems, seed 42)

| Metric | Haiku 4.5 | GPT-4.1 mini | GPT-4.1 |
|---|---|---|---|
| **Passed all** | 59 | 55 | **66** |
| Wrong | 35 | 45 | **32** |
| Failed to run | 6 | **0** | 2 |
| Introductory (of 17) | **15** | **15** | 14 |
| Interview (of 68) | 37 | 35 | **43** |
| Competition (of 15) | 7 | 5 | **9** |
| Output tokens | 118,659 | 42,278 | **41,647** |
| Approx. real cost | $0.66 | **$0.09** | $0.46 |

**GPT-4.1** is the strongest here (66 passed), with the biggest lead on the
harder interview/competition tiers. **Haiku 4.5** is second overall (59) but
top on introductory problems, at the cost of the most unrunnable programs (6)
and ~2.8× the output tokens of either GPT model. **GPT-4.1 mini** is the
cheapest (~$0.09) and never produced unrunnable code, but solved the fewest.

## Reproduce

```bash
cd saturation_test
export HF_DATASETS_TRUST_REMOTE_CODE=1
export OPENROUTER_API_KEY=sk-or-...            # an OpenRouter key with budget

# Haiku 4.5 (defaults)
python3 -u run_eval.py > run_haiku-4.5.log 2>&1

# Any other OpenRouter model
EVAL_MODEL='openai/gpt-4.1-mini' EVAL_TAG='gpt-4.1-mini' \
  python3 -u run_eval.py > run_gpt-4.1-mini.log 2>&1
```

Knobs live at the top of `run_eval.py`: `N`, `SEED`, `MAX_TOKENS`,
`PER_TEST_TIMEOUT`, `MAX_TESTS_PER_PROBLEM`, `WORKERS`.

## Files

| File | Contents |
|---|---|
| `run_eval.py` | Sampling, prompting, OpenRouter calls, grading harness. |
| `results_haiku-4.5.json` | Per-problem records (mode, status, generated code, pass counts, tokens) + summary for Haiku 4.5. |
| `results_gpt-4.1-mini.json` | Same, for GPT-4.1 mini (produced by the launched run). |
| `run_haiku-4.5.log` / `run_gpt-4.1-mini.log` | Live progress logs (one line per 5 completed problems). |
| `EXPERIMENT.md` | This document. |

## Notes & limitations

- **20-test cap** (above) makes `passed_all` an upper bound on true strict accuracy.
- The grader executes model-generated code with timeouts but **is not a hardened
  sandbox** — run on a throwaway/CI machine.
- Output comparison normalizes whitespace and allows `1e-6` float tolerance;
  exotic judge formats (special checkers, multiple valid answers) may be
  mis-scored as `wrong`.
- Single sample per problem at `temperature = 0` (APPS-style strict accuracy,
  not pass@k).
- `temperature = 0` and `seed = 42` make the sampling and (approximately) the
  generations reproducible; provider-side nondeterminism can still cause minor
  drift.
