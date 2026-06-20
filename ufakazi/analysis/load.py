"""Read eval logs into a tidy frame and compute the bias results.

Everything runs off Inspect's `samples_df`, one row per (trial, epoch). The scorer's
choice logprob lives in score metadata, so we pull it in explicitly with a column spec.

The result design rests on the counterbalancing in `trials.py`:

- **same-language controls** (both testimonies one language) expose pure position and
  content bias, with no language difference: the baseline.
- **cross-language trials** clothe the two contents in different languages. Aggregated
  symmetrically over which content is in which language, content bias cancels and what
  remains is the language main effect: `P(prefer the <target>-written testimony)` vs 0.5.

Significance uses a **scenario-level (cluster) bootstrap**: scenarios are the unit of
independence, so we resample scenarios with replacement and recompute, which respects the
nesting of epochs within trial within scenario without assuming independent rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from inspect_ai.analysis import SampleColumn, SampleSummary, samples_df

CHOICE_COLUMN = "score_record_choice"
VALID_CHOICES = ("A", "B")
CONTROL = "same_language_control"
CROSS = "cross_language"
DEFAULT_REFERENCE = "en"


def _column_spec() -> list:
    """Default sample columns plus the scorer's logprob, rationale, and raw completion."""
    return list(SampleSummary) + [
        SampleColumn(
            "choice_logprob", path="scores.record_choice.metadata.choice_logprob"
        ),
        SampleColumn(
            "chosen_position", path="scores.record_choice.metadata.chosen_position"
        ),
        SampleColumn("rationale", path="scores.record_choice.explanation"),
        SampleColumn("completion", path="scores.record_choice.answer"),
    ]


def load_trials(logs: str = "results/logs") -> pd.DataFrame:
    """Load trials (one row per trial-epoch) with derived content-language columns.

    Adds `valid` (choice parsed), `chose_first` (position baseline), `lang_A`/`lang_B`
    (language assigned to each *content*, independent of presentation order), and
    `lang_chosen` (language of the content the model picked)."""
    df = samples_df(logs, columns=_column_spec())
    df = df.rename(columns={CHOICE_COLUMN: "chosen_id"})

    df["valid"] = df["chosen_id"].isin(VALID_CHOICES)
    df["chose_first"] = df["chosen_id"] == df["metadata_first_testimony_id"]

    first_is_a = df["metadata_first_testimony_id"] == "A"
    df["lang_A"] = np.where(
        first_is_a, df["metadata_lang_first"], df["metadata_lang_second"]
    )
    df["lang_B"] = np.where(
        first_is_a, df["metadata_lang_second"], df["metadata_lang_first"]
    )
    chose_a = df["chosen_id"] == "A"
    df["lang_chosen"] = np.where(
        df["valid"], np.where(chose_a, df["lang_A"], df["lang_B"]), None
    )
    return df


def filter_latest_eval(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the most recent eval run in the frame.

    `samples_df` aggregates every `.eval` log in the directory; mixing runs with different
    language sets or models would corrupt the aggregates. Log filenames are timestamped, so
    the lexicographically-last `log` identifies the latest run."""
    if df.empty or "eval_id" not in df.columns or "log" not in df.columns:
        return df
    latest_eval_id = df.sort_values("log")["eval_id"].iloc[-1]
    return df[df["eval_id"] == latest_eval_id]


def _pair_cross_trials(df: pd.DataFrame, target: str, reference: str) -> pd.DataFrame:
    """Valid cross-language trials whose two content languages are exactly {target, ref}."""
    pair = {target, reference}
    cross = df[(df["metadata_condition"] == CROSS) & df["valid"]].copy()
    if cross.empty:
        return cross
    keep = cross.apply(lambda r: {r["lang_A"], r["lang_B"]} == pair, axis=1)
    return cross[keep]


def _cluster_bootstrap_ci(
    values_by_scenario: dict[str, np.ndarray],
    n_boot: int,
    seed: int,
    ci: float = 95.0,
) -> tuple[float, float]:
    """Percentile CI for the grand mean, resampling whole scenarios with replacement."""
    scenarios = list(values_by_scenario)
    if not scenarios:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(scenarios)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        picks = rng.integers(0, n, n)
        vals = np.concatenate([values_by_scenario[scenarios[j]] for j in picks])
        stats[i] = vals.mean()
    lo, hi = np.percentile(stats, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(lo), float(hi)


def _continuous_target_prob(cross: pd.DataFrame, target: str) -> float | None:
    """Mean probability mass the model put on the target-language content per trial.

    From the chosen-token logprob: if it chose the target, that prob *is* P(target); if it
    chose the reference, P(target) = 1 - P(chosen). None when no logprobs are present."""
    sub = cross[cross["choice_logprob"].notna()]
    if sub.empty:
        return None
    p_chosen = np.exp(sub["choice_logprob"].astype(float).to_numpy())
    chose_target = (sub["lang_chosen"] == target).to_numpy()
    p_target = np.where(chose_target, p_chosen, 1.0 - p_chosen)
    return float(p_target.mean())


def language_preference(
    df: pd.DataFrame,
    target: str,
    reference: str = DEFAULT_REFERENCE,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Language main effect: P(prefer the target-written testimony) in {target, ref} cross
    trials, with a scenario-bootstrap 95% CI. CI excluding 0.5 => significant preference."""
    cross = _pair_cross_trials(df, target, reference)
    if cross.empty:
        return {
            "target": target,
            "reference": reference,
            "n_cross_trials": 0,
            "p_prefer_target": float("nan"),
            "ci95": (float("nan"), float("nan")),
            "significant": False,
            "p_prefer_target_continuous": None,
        }
    chose_target = (cross["lang_chosen"] == target).astype(float)
    by_scenario = {
        s: chose_target[cross["metadata_scenario_id"] == s].to_numpy()
        for s in cross["metadata_scenario_id"].unique()
    }
    lo, hi = _cluster_bootstrap_ci(by_scenario, n_boot, seed)
    return {
        "target": target,
        "reference": reference,
        "n_cross_trials": int(len(cross)),
        "p_prefer_target": float(chose_target.mean()),
        "ci95": (lo, hi),
        "significant": not (lo <= 0.5 <= hi),
        "p_prefer_target_continuous": _continuous_target_prob(cross, target),
    }


def per_scenario_language_effect(
    df: pd.DataFrame, target: str, reference: str = DEFAULT_REFERENCE
) -> pd.DataFrame:
    """Per-scenario language effect, content held constant:
    `P(chose A | A=target) - P(chose A | A=reference)`. Positive => the target language
    lifts a testimony's credibility. Surfaces which scenarios drive the main effect
    (e.g. the deliberately length-imbalanced ones)."""
    cross = _pair_cross_trials(df, target, reference)
    rows = []
    if not cross.empty:
        cross = cross.assign(chose_A=(cross["chosen_id"] == "A").astype(float))
        for scenario_id, g in cross.groupby("metadata_scenario_id"):
            a_is_target = g.loc[g["lang_A"] == target, "chose_A"]
            a_is_reference = g.loc[g["lang_A"] == reference, "chose_A"]
            if len(a_is_target) and len(a_is_reference):
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "language_effect": float(
                            a_is_target.mean() - a_is_reference.mean()
                        ),
                        "n_trials": int(len(g)),
                    }
                )
    return pd.DataFrame(rows).sort_values("language_effect", ignore_index=True)


def control_baselines(df: pd.DataFrame) -> dict:
    """Same-language-control baselines: position bias (~0.5 = none) and residual content
    preference for A (~0.5 once position is counterbalanced)."""
    ctrl = df[(df["metadata_condition"] == CONTROL) & df["valid"]]
    n = len(ctrl)
    return {
        "n_control_trials": int((df["metadata_condition"] == CONTROL).sum()),
        "position_first_rate": float(ctrl["chose_first"].mean()) if n else float("nan"),
        "content_pref_A_rate": float((ctrl["chosen_id"] == "A").mean())
        if n
        else float("nan"),
    }


def summarize(df: pd.DataFrame) -> dict:
    """Parse health plus the control baselines (back-compat shape for the CLI)."""
    return {
        "n_trials": int(len(df)),
        "n_parse_errors": int((~df["valid"]).sum()),
        **control_baselines(df),
    }


def language_report(
    df: pd.DataFrame,
    reference: str = DEFAULT_REFERENCE,
    n_boot: int = 2000,
    seed: int = 0,
) -> list[dict]:
    """Language preference vs the reference for every other language seen in cross trials."""
    cross = df[df["metadata_condition"] == CROSS]
    targets = sorted(
        set(cross["lang_A"]).union(cross["lang_B"]) - {reference}
        if not cross.empty
        else set()
    )
    return [
        language_preference(df, target, reference, n_boot=n_boot, seed=seed)
        for target in targets
    ]
