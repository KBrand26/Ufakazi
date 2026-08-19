"""The final 9-model panel, loaded the one correct way.

Two traps make the panel logs unsuitable for `filter_latest_per_model`:

1. Gemini 3.5 Flash's run was interrupted and resumed, so its trials span two `.eval`
   logs in `results/logs_b1`. Keeping only the latest eval per model silently drops most
   of Gemini. The panel dir must be **pooled** per model instead (every eval, additive).
2. GPT-4o-mini's panel run (10 epochs) lives in `results/logs`, alongside older
   smoke-test evals of the same model, so *there* the latest eval is the right one.

`load_panel` applies both rules and drops the keyless mock. It reproduces the paper's
Table 1 exactly; the figures and the rationale analysis both read from it.
"""

from __future__ import annotations

import pandas as pd

from ufakazi.analysis.load import filter_latest_eval, load_trials

PANEL_LOG_DIR = "results/logs_b1"
MINI_LOG_DIR = "results/logs"
MINI_MODEL = "openrouter/openai/gpt-4o-mini"
MOCK_MARKER = "mockllm"


def load_panel(
    panel_dir: str = PANEL_LOG_DIR,
    mini_dir: str | None = MINI_LOG_DIR,
    mini_model: str = MINI_MODEL,
) -> pd.DataFrame:
    """All panel trials: `panel_dir` pooled per model (mock excluded) plus, if `mini_dir`
    is given, the latest eval of `mini_model` found there."""
    panel = load_trials(panel_dir)
    panel = panel[~panel["model"].astype(str).str.contains(MOCK_MARKER)]
    if mini_dir is None:
        return panel.reset_index(drop=True)
    mini = load_trials(mini_dir)
    mini = filter_latest_eval(mini[mini["model"] == mini_model])
    return pd.concat([panel, mini], ignore_index=True)
