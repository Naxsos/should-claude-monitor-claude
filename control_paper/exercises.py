# %%

import os
import re
import sys
import warnings
from pathlib import Path
from pprint import pprint
from typing import Any, Literal, Optional, Union

from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, scorer
from inspect_ai.scorer._metrics import accuracy, stderr
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver
from inspect_ai.util import ExecResult, sandbox

# Make sure exercises are in the path
section = "part5_ai_control"
# The part5_ai_control package lives alongside this file in control_paper/.
if "__file__" in globals():
    exercises_dir = Path(__file__).resolve().parent
else:
    # Interactive (# %%) execution: __file__ is undefined, so locate control_paper/
    # (which holds the part5_ai_control package) relative to the current dir.
    _cands = [c / section for c in [Path.cwd(), *Path.cwd().parents]] + [
        p / "control_paper" / section
        for c in [Path.cwd(), *Path.cwd().parents]
        for p in c.rglob("should-claude-monitor-claude")
    ]
    exercises_dir = next(c.parent for c in _cands if c.exists())
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
os.chdir(exercises_dir)
import part5_ai_control.tests as tests
import part5_ai_control.utils as utils
from part5_ai_control.utils import (
    analyze_detector_scores,
    assemble_protocol_data,
    between,
    latest_eval_log,
    normalize_content,
    parse_main_output,
    parse_trigger,
    plot_protocol_comparison,
    plot_safety_usefulness_pareto,
    plot_sus_distribution,
    run_all_protocols,
)

cleaned_apps = hf_hub_download(
    repo_id="styme3279/control-apps-cleaned",
    filename="cleaned_apps.jsonl",
    repo_type="dataset",
)

MAIN = __name__ == "__main__"

# Run flags 

# Model constants
UNTRUSTED_MODEL = "openrouter/anthropic/claude-sonnet-4.5"
TRUSTED_MODEL = "openrouter/google/gemini-2.5-flash"

# Sandbox constants
USING_DOCKER = True  # set False on Runpod / Vast / Lambda where Docker isn't available
docker_exec = "/usr/bin/python3"
local_exec = "python3"

# Defaults
DEFAULT_N = 5

load_dotenv()

assert os.getenv("OPENROUTER_API_KEY") is not None, (
    "Set OPENROUTER_API_KEY in a .env file (or Colab secrets) — see Inspect docs for details."
)


# %%

