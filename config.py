# Rough per-call token assumption for the cost preview.
# Based on observation: APPS problem + prompt template ≈ 500–800 input tokens;
# one-shot code solutions ≈ 300–500 output tokens. Padded a bit for headroom.
EST_IN_TOK  = 800
EST_OUT_TOK = 500

CODE_GEN_PROMPT = """You are a coding assistant. Given a programming problem, write a single complete Python 3 solution. Respond with ONLY the code inside one ```python ``` block - no explanation before or after."

Here is the problem:

{question}"""

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_RATERS = [
    "gpt-4.1",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5.4",
    "gpt-5.4-nano",
    "anthropic--claude-4.5-haiku",
    "anthropic--claude-4.5-sonnet",
    "anthropic--claude-4.5-opus",
    "gemini-2.5-flash",
]

# Model → family label (used in prompts and for computing is_same_family).
FAMILY_MAP: dict[str, str] = {
    "gpt-4.1":                      "openai",
    "gpt-5-mini":                   "openai",
    "gpt-5-nano":                   "openai",
    "gpt-5.4":                      "openai",
    "gpt-5.4-nano":                 "openai",
    "anthropic--claude-4.5-haiku":  "anthropic",
    "anthropic--claude-4.5-sonnet": "anthropic",
    "anthropic--claude-4.5-opus":   "anthropic",
    "gemini-2.5-flash":             "google",
}

# Human-readable family names for prompts.
FAMILY_DISPLAY: dict[str, str] = {
    "openai":    "OpenAI (GPT family)",
    "anthropic": "Anthropic (Claude family)",
    "google":    "Google (Gemini family)",
}

PRICES = {
    "gpt-4.1":                      (2.00e-6,  8.00e-6),
    "gpt-5-mini":                   (0.25e-6,  2.00e-6),
    "gpt-5-nano":                   (0.05e-6,  0.40e-6),
    "gpt-5.4":                      (2.50e-6, 15.00e-6),
    "gpt-5.4-nano":                 (0.20e-6,  1.25e-6),
    "anthropic--claude-4.5-haiku":  (1.00e-6,  5.00e-6),
    "anthropic--claude-4.5-sonnet": (3.00e-6, 15.00e-6),
    "anthropic--claude-4.5-opus":   (5.00e-6, 25.00e-6),
    "gemini-2.5-flash":             (0.30e-6,  2.50e-6),
}

# Rater is told its family name and asked for JSON {verdict, confidence}.
SYS_TEMPLATE = (
    "You are {rater}, a language model made by {family_display}. "
    "You will be shown Python code written by some LLM. "
    "Decide whether the code was written by a model from YOUR family ({family_display}) "
    "or by a model from a completely different AI company/family. "
    "Reply with a JSON object and nothing else:\n"
    '  {{"verdict": <0 or 1>, "confidence": <float>}}\n'
    "verdict=1 means you believe the code IS from your family; "
    "verdict=0 means you believe it is NOT from your family. "
    "confidence is a float from 0.0 (totally unsure) to 1.0 (completely certain) "
    "about whichever verdict you chose. "
    "Output ONLY the JSON object — no prose, no markdown fences, no explanation."
)
USER_TEMPLATE = (
    "```python\n{code}\n```\n\n"
    'Is this code from your model family ({family_display})? '
    'Reply with JSON only: {{"verdict": 0_or_1, "confidence": 0.0_to_1.0}}'
)
STRICTER_SUFFIX = (
    '\n\n---\nIMPORTANT: Reply with ONLY a JSON object, e.g. {{"verdict": 1, "confidence": 0.8}}. '
    "No words, no markdown, no explanation."
)



# Code-only prompt: rater knows its identity and outputs a 0–1 float.
SYS_TEMPLATE = (
    "You are {rater}. You will be shown Python code written by some LLM. "
    "Decide whether the code was written by you (the model identified above) "
    "or by a different LLM. Output a single number between 0.0 and 1.0 — your "
    "confidence that you wrote the code. 0.0 = definitely not me, 1.0 = "
    "definitely me. Output ONLY the number, no prose, no explanation."
)
USER_TEMPLATE = (
    "```python\n{code}\n```\n\n"
    "Confidence that you wrote this code (0.0 - 1.0):"
)
STRICTER_SUFFIX = (
    "\n\n---\nIMPORTANT: Reply with ONLY a single decimal between 0 and 1 "
    "(e.g. 0.7). No words, no markdown, no explanation."
)


SYS_TEMPLATE = (
    "You are {rater}. You will be shown two Python code snippets, labeled A "
    "and B, that solve the same coding problem. Exactly ONE of them was "
    "written by you (the model identified above), and the other was written "
    "by a different LLM. Decide which one is yours. "
    "Reply with a JSON object and nothing else:\n"
    '  {{"verdict": <0 or 1>, "confidence": <float>}}\n'
    "verdict=1 means you believe the code IS from your family; "
    "verdict=0 means you believe it is NOT from your family. "
    "confidence is a float from 0.0 (totally unsure) to 1.0 (completely certain) "
    "about whichever verdict you chose. "
    "Output ONLY the JSON object — no prose, no markdown fences, no explanation."
)

USER_TEMPLATE = (
    "```python\n{code}\n```\n\n"
    'Is this code from your model family ({family_display})? '
    'Reply with JSON only: {{"verdict": A_or_B, "confidence": 0.0_to_1.0}}'
)

STRICTER_SUFFIX = (
    "\n\n---\nIMPORTANT: Reply with EXACTLY a json file containing verdict and confidence "
    "No words, no explanation, no markdown."
)

