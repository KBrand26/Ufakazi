"""Read eval logs into a tidy frame and compute the M1 baselines.

Everything here runs off Inspect's `samples_df`, which flattens one row per trial with
sample metadata expanded as `metadata_*` columns and the scorer value as
`score_record_choice`. The position baseline and content preference are derived purely
from those default columns, so no custom column spec is needed yet.
"""

from __future__ import annotations

import pandas as pd
from inspect_ai.analysis import samples_df

CHOICE_COLUMN = "score_record_choice"
VALID_CHOICES = ("A", "B")


def load_trials(logs: str = "results/logs") -> pd.DataFrame:
    """Load trials and add a `chose_first` flag (chosen content == first-presented)."""
    df = samples_df(logs)
    if CHOICE_COLUMN in df.columns:
        df = df.rename(columns={CHOICE_COLUMN: "chosen_id"})
    df["chose_first"] = df["chosen_id"] == df["metadata_first_testimony_id"]
    return df


def summarize(df: pd.DataFrame) -> dict:
    """M1 baselines: parse health, the position bias the control should expose, and the
    content preference (which should sit near 0.5 once position is counterbalanced)."""
    valid = df[df["chosen_id"].isin(VALID_CHOICES)]
    n_valid = len(valid)
    return {
        "n_trials": len(df),
        "n_parse_errors": int(len(df) - n_valid),
        "position_first_rate": float(valid["chose_first"].mean())
        if n_valid
        else float("nan"),
        "content_pref_A_rate": float((valid["chosen_id"] == "A").mean())
        if n_valid
        else float("nan"),
    }
