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
from inspect_ai.analysis import EvalColumn, SampleColumn, SampleSummary, samples_df

CHOICE_COLUMN = "score_record_choice"
VALID_CHOICES = ("A", "B")
CONTROL = "same_language_control"
CROSS = "cross_language"
DEFAULT_REFERENCE = "en"
MT_SUFFIX = "_mt"

# A scenario is "saturated" for a model when its control content_skew (|P(chose A) - 0.5|)
# is at least this high, i.e. the model picks one testimony >= 90% of the time regardless of
# position. Saturation is judged per model from its own controls, not tagged in the data, so
# a scenario a strong model balances is not mislabelled by a weaker model's preference.
SATURATION_THRESHOLD = 0.4


def _column_spec() -> list:
    """Default sample columns plus the eval's model, the scorer's logprob, rationale, and
    raw completion. The `model` column is what makes the frame model-aware: a multi-model
    sweep writes one eval (one model) per `.eval` log, so without it the per-model results
    would be pooled into nonsense."""
    return (
        list(SampleSummary)
        + [EvalColumn("model", path="eval.model")]
        + [
            SampleColumn(
                "choice_logprob", path="scores.record_choice.metadata.choice_logprob"
            ),
            SampleColumn(
                "chosen_position", path="scores.record_choice.metadata.chosen_position"
            ),
            SampleColumn("rationale", path="scores.record_choice.explanation"),
            SampleColumn("completion", path="scores.record_choice.answer"),
        ]
    )


def load_trials(logs: str = "results/logs") -> pd.DataFrame:
    """Load trials (one row per trial-epoch) with derived content-language columns.

    Adds `valid` (choice parsed), `chose_first` (position baseline), `lang_A`/`lang_B`
    (language assigned to each *content*, independent of presentation order), and
    `lang_chosen` (language of the content the model picked)."""
    return _derive_trial_columns(samples_df(logs, columns=_column_spec()))


def _derive_trial_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add `valid` / `chose_first` / `lang_A` / `lang_B` / `lang_chosen` to a raw samples
    frame. Split out from `load_trials` so the NA-handling is unit-testable without disk."""
    df = df.rename(columns={CHOICE_COLUMN: "chosen_id"})

    # Error/interrupted samples (e.g. a run that hit an API limit mid-stream) land in the
    # frame with NA chosen_id and metadata. Mask comparisons must stay NA-free or np.where
    # raises "boolean value of NA is ambiguous"; `fillna(False)` drops those rows out of every
    # derived column, and `valid` already excludes them downstream.
    df["valid"] = df["chosen_id"].isin(VALID_CHOICES)
    df["chose_first"] = (df["chosen_id"] == df["metadata_first_testimony_id"]).fillna(
        False
    )

    first_is_a = (df["metadata_first_testimony_id"] == "A").fillna(False)
    df["lang_A"] = np.where(
        first_is_a, df["metadata_lang_first"], df["metadata_lang_second"]
    )
    df["lang_B"] = np.where(
        first_is_a, df["metadata_lang_second"], df["metadata_lang_first"]
    )
    chose_a = (df["chosen_id"] == "A").fillna(False)
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


def filter_latest_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """Keep each model's most recent eval run, dropping older runs of the same model.

    A multi-model sweep writes one `.eval` per model, so the whole batch survives (one
    eval each); re-running a model supersedes its earlier eval rather than pooling into it.
    This is the per-model generalization of `filter_latest_eval`: the right default when a
    log dir holds a full panel plus stale re-runs. To instead pool every run of a model
    (additive epochs), skip this and analyze the frame directly."""
    if df.empty or not {"model", "eval_id", "log"} <= set(df.columns):
        return df
    order = df.sort_values("log")
    latest = order.groupby("model")["eval_id"].last()
    keep = set(latest.values)
    return df[df["eval_id"].isin(keep)]


def models_in(df: pd.DataFrame) -> list[str]:
    """Distinct models present, in stable (sorted) order."""
    if "model" not in df.columns:
        return []
    return sorted(m for m in df["model"].dropna().unique())


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
    cols = ["scenario_id", "language_effect", "n_trials"]
    if (
        not rows
    ):  # no scenario has both A=target and A=reference cross trials (sparse data)
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).sort_values(
        "language_effect", ignore_index=True
    )


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


def per_scenario_control_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Per-scenario content balance from the same-language controls, a scenario-quality
    diagnostic. In a control both testimonies share one language, so over the two position
    orders (which cancel) `P(chose A)` is pure content preference: `content_skew` =
    `|P(chose A) - 0.5|` is how lopsided the scenario's two testimonies are.

    A near-0.5 skew (P(chose A) ~ 0 or 1) flags a *saturated* scenario whose content has a
    runaway favourite. Counterbalancing means this does not bias the language main effect,
    but a saturated scenario contributes little signal: the choice is pinned at the floor or
    ceiling, so language has no headroom to move it. Sorted worst (most skewed) first."""
    ctrl = df[(df["metadata_condition"] == CONTROL) & df["valid"]]
    rows = [
        {
            "scenario_id": sid,
            "p_chose_A": float((g["chosen_id"] == "A").mean()),
            "content_skew": abs(float((g["chosen_id"] == "A").mean()) - 0.5),
            "n_control": int(len(g)),
        }
        for sid, g in ctrl.groupby("metadata_scenario_id")
    ]
    cols = ["scenario_id", "p_chose_A", "content_skew", "n_control"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).sort_values(
        "content_skew", ascending=False, ignore_index=True
    )


def saturation_labels(
    df: pd.DataFrame, threshold: float = SATURATION_THRESHOLD
) -> pd.DataFrame:
    """Per-scenario control balance plus a `saturated` flag and the `favoured_id` (the
    testimony the model picks in the control). The split key for the two analysis arms:
    `saturated == False` scenarios feed the language main effect (content balanced, the
    counterbalancing assumption holds); `saturated == True` scenarios feed the override
    probe. Pass a single model's frame, so saturation is judged per model."""
    bal = per_scenario_control_balance(df)
    if bal.empty:
        return bal.assign(
            saturated=pd.Series(dtype=bool), favoured_id=pd.Series(dtype=str)
        )
    bal = bal.copy()
    bal["saturated"] = bal["content_skew"] >= threshold
    bal["favoured_id"] = np.where(bal["p_chose_A"] >= 0.5, "A", "B")
    return bal


def content_override(
    df: pd.DataFrame,
    reference: str = DEFAULT_REFERENCE,
    threshold: float = SATURATION_THRESHOLD,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Override probe on the saturated scenarios: can language flip a content preference?

    For each saturated scenario the control fixes the model's *content* favourite. In the
    cross-language trials that content is sometimes written in the reference language and
    sometimes in a target language. We measure `P(still pick the favoured testimony)` in each
    case: when the favourite is written in the reference it should stay strongly preferred
    (the within-pair baseline); if writing it in the target language drops that below 0.5,
    language has *overridden* a content preference the model otherwise holds.

    One row per target language: the two probabilities, the drop between them (the override
    magnitude), a scenario-bootstrap CI on the target-language probability, and how many of
    the saturated scenarios actually flipped (per-scenario mean < 0.5). Pass one model's
    frame. Empty if no scenario is saturated for this model."""
    labels = saturation_labels(df, threshold)
    sat = set(labels.loc[labels["saturated"], "scenario_id"])
    favoured = dict(zip(labels["scenario_id"], labels["favoured_id"]))
    cols = [
        "target",
        "p_pick_favoured_in_reference",
        "p_pick_favoured_in_target",
        "override_drop",
        "ci_lo",
        "ci_hi",
        "n_scenarios",
        "n_flipped",
    ]
    cross = df[(df["metadata_condition"] == CROSS) & df["valid"]].copy()
    cross = cross[cross["metadata_scenario_id"].isin(sat)]
    if cross.empty:
        return pd.DataFrame(columns=cols)

    cross["favoured_id"] = cross["metadata_scenario_id"].map(favoured)
    cross["chose_favoured"] = (cross["chosen_id"] == cross["favoured_id"]).astype(float)
    cross["lang_favoured"] = np.where(
        cross["favoured_id"] == "A", cross["lang_A"], cross["lang_B"]
    )

    rows = []
    targets = sorted((set(cross["lang_A"]) | set(cross["lang_B"])) - {reference, None})
    for target in targets:
        pair = cross[
            cross.apply(
                lambda r: {r["lang_A"], r["lang_B"]} == {reference, target}, axis=1
            )
        ]
        in_target = pair[pair["lang_favoured"] == target]
        in_reference = pair[pair["lang_favoured"] == reference]
        if in_target.empty:
            continue
        by_scenario = {
            s: in_target.loc[
                in_target["metadata_scenario_id"] == s, "chose_favoured"
            ].to_numpy()
            for s in in_target["metadata_scenario_id"].unique()
        }
        per_scenario_mean = {s: v.mean() for s, v in by_scenario.items()}
        lo, hi = _cluster_bootstrap_ci(by_scenario, n_boot, seed)
        p_target = float(in_target["chose_favoured"].mean())
        p_reference = (
            float(in_reference["chose_favoured"].mean())
            if not in_reference.empty
            else float("nan")
        )
        rows.append(
            {
                "target": target,
                "p_pick_favoured_in_reference": p_reference,
                "p_pick_favoured_in_target": p_target,
                "override_drop": p_reference - p_target,
                "ci_lo": lo,
                "ci_hi": hi,
                "n_scenarios": len(by_scenario),
                "n_flipped": int(sum(m < 0.5 for m in per_scenario_mean.values())),
            }
        )
    return pd.DataFrame(rows, columns=cols)


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


# --- Tidy per-model result tables (the inputs the paper figures + LaTeX macros consume) ---


def provenance_pairs(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Human/machine sibling pairs present: a `{x}_mt` machine code whose `{x}` human base
    is also assigned somewhere in the frame (e.g. `(afr, afr_mt)`). These isolate the
    translation-source effect with content and language held constant."""
    langs = set(df["lang_A"].dropna()) | set(df["lang_B"].dropna())
    pairs = [
        (code[: -len(MT_SUFFIX)], code)
        for code in sorted(langs)
        if code.endswith(MT_SUFFIX) and code[: -len(MT_SUFFIX)] in langs
    ]
    return pairs


def _flatten_pref(model: str, pref: dict) -> dict:
    """One `language_preference` dict flattened to a tidy row, CI split into lo/hi."""
    lo, hi = pref["ci95"]
    return {
        "model": model,
        "reference": pref["reference"],
        "target": pref["target"],
        "p_prefer_target": pref["p_prefer_target"],
        "ci_lo": lo,
        "ci_hi": hi,
        "significant": pref["significant"],
        "n_cross_trials": pref["n_cross_trials"],
        "p_prefer_target_continuous": pref["p_prefer_target_continuous"],
    }


def _balanced_subset(mdf: pd.DataFrame) -> pd.DataFrame:
    """A single model's trials restricted to its balanced (non-saturated) scenarios, the
    clean estimand for the language main effect. With no control trials (e.g. synthetic test
    frames) saturation can't be judged, so the frame is returned unfiltered."""
    labels = saturation_labels(mdf)
    if labels.empty:
        return mdf
    balanced = set(labels.loc[~labels["saturated"], "scenario_id"])
    return mdf[mdf["metadata_scenario_id"].isin(balanced)]


def language_effect_table(
    df: pd.DataFrame,
    reference: str = DEFAULT_REFERENCE,
    n_boot: int = 2000,
    seed: int = 0,
    *,
    balanced_only: bool = True,
) -> pd.DataFrame:
    """Per-model language main effect: one row per (model, target language) with the
    preference, its scenario-bootstrap 95% CI, significance and n. Drives the forest plot,
    the model x language heatmap, and the cross-linguistic gradient.

    `balanced_only` (default) restricts each model to its balanced scenarios, the clean
    main-effect arm; the saturated scenarios are reported separately by
    `override_effect_table`. Saturation is judged per model from its own controls."""
    rows = []
    for model in models_in(df):
        mdf = df[df["model"] == model]
        if balanced_only:
            mdf = _balanced_subset(mdf)
        rows.extend(
            _flatten_pref(model, pref)
            for pref in language_report(mdf, reference, n_boot=n_boot, seed=seed)
        )
    return pd.DataFrame(
        rows,
        columns=[
            "model",
            "reference",
            "target",
            "p_prefer_target",
            "ci_lo",
            "ci_hi",
            "significant",
            "n_cross_trials",
            "p_prefer_target_continuous",
        ],
    )


def override_effect_table(
    df: pd.DataFrame,
    reference: str = DEFAULT_REFERENCE,
    threshold: float = SATURATION_THRESHOLD,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-model content-override probe, one row per (model, target language): how the
    model's content favourite survives being written in the target language, plus the
    fraction of saturated scenarios where the favourite flipped. Drives the override figure;
    the saturated-arm companion to `language_effect_table`."""
    rows = []
    for model in models_in(df):
        over = content_override(
            df[df["model"] == model], reference, threshold, n_boot=n_boot, seed=seed
        )
        for _, r in over.iterrows():
            n = r["n_scenarios"]
            rows.append(
                {
                    "model": model,
                    "target": r["target"],
                    "p_pick_favoured_in_target": r["p_pick_favoured_in_target"],
                    "p_pick_favoured_in_reference": r["p_pick_favoured_in_reference"],
                    "override_drop": r["override_drop"],
                    "flip_fraction": (r["n_flipped"] / n) if n else float("nan"),
                    "ci_lo": r["ci_lo"],
                    "ci_hi": r["ci_hi"],
                    "n_scenarios": int(n),
                    "n_flipped": int(r["n_flipped"]),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "model",
            "target",
            "p_pick_favoured_in_target",
            "p_pick_favoured_in_reference",
            "override_drop",
            "flip_fraction",
            "ci_lo",
            "ci_hi",
            "n_scenarios",
            "n_flipped",
        ],
    )


def provenance_effect_table(
    df: pd.DataFrame, n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Per-model human-vs-machine effect for each provenance sibling pair: P(prefer the
    machine translation over the human one), same content and language. Drives the
    translation-source figure that bounds confidence in the MT-only languages."""
    pairs = provenance_pairs(df)
    rows = []
    for model in models_in(df):
        mdf = df[df["model"] == model]
        for human, machine in pairs:
            pref = language_preference(
                mdf, target=machine, reference=human, n_boot=n_boot, seed=seed
            )
            row = _flatten_pref(model, pref)
            row["human"] = human
            row["machine"] = machine
            rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            "model",
            "human",
            "machine",
            "reference",
            "target",
            "p_prefer_target",
            "ci_lo",
            "ci_hi",
            "significant",
            "n_cross_trials",
            "p_prefer_target_continuous",
        ],
    )


def control_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model control diagnostics: parse health and the same-language-control baselines
    (position bias, residual content preference), each ~0.5 if counterbalancing held."""
    rows = []
    for model in models_in(df):
        mdf = df[df["model"] == model]
        rows.append({"model": model, **summarize(mdf)})
    return pd.DataFrame(
        rows,
        columns=[
            "model",
            "n_trials",
            "n_parse_errors",
            "n_control_trials",
            "position_first_rate",
            "content_pref_A_rate",
        ],
    )
