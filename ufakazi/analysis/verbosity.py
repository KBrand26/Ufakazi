"""Is the language effect actually a verbosity effect?

The worry: Afrikaans testimonies may be longer (translation tends to inflate length), so a
model that simply favours shorter testimonies would look like it favours English. This
module measures length directly and, crucially, isolates a **language-free** length bias
from the same-language controls (both testimonies one language, so any length difference is
pure content). If the model prefers shorter testimonies even within controls, length is a
real confound for the headline language effect.

Lengths are read from the scenario fixtures (the exact text shown), joined to the logged
trials by (scenario, testimony, language), so no re-run is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ufakazi.analysis.load import (
    CONTROL,
    DEFAULT_REFERENCE,
    _cluster_bootstrap_ci,
    _pair_cross_trials,
)
from ufakazi.scenarios.loader import load_scenarios


def _length(text: str, measure: str) -> int:
    return len(text) if measure == "chars" else len(text.split())


def length_lookup(measure: str = "chars") -> dict[tuple[str, str, str], int]:
    """(scenario_id, testimony_id, language) -> testimony length."""
    out: dict[tuple[str, str, str], int] = {}
    for scenario in load_scenarios():
        for testimony in scenario.testimonies:
            for lang, tr in testimony.translations.items():
                out[(scenario.scenario_id, testimony.testimony_id, lang)] = _length(
                    tr.text, measure
                )
    return out


def length_summary(
    target: str = "afr", reference: str = DEFAULT_REFERENCE, measure: str = "chars"
) -> dict:
    """Per-language mean length and the target/reference ratio, content-matched: tests the
    'is Afrikaans longer?' premise. Ratio > 1 means the target language runs longer."""
    pairs = []
    for scenario in load_scenarios():
        for testimony in scenario.testimonies:
            tr = testimony.translations
            if target in tr and reference in tr:
                pairs.append(
                    (
                        _length(tr[reference].text, measure),
                        _length(tr[target].text, measure),
                    )
                )
    if not pairs:
        return {}
    ref_lens, tgt_lens = zip(*pairs)
    ratios = [t / r for r, t in pairs if r]
    return {
        "measure": measure,
        "n_testimonies": len(pairs),
        f"mean_len_{reference}": float(np.mean(ref_lens)),
        f"mean_len_{target}": float(np.mean(tgt_lens)),
        "mean_ratio_target_over_reference": float(np.mean(ratios)),
        "target_longer_fraction": float(np.mean([t > r for r, t in pairs])),
    }


def length_ratio_table(
    targets: tuple[str, ...] = ("afr", "afr_mt", "zul_mt", "xho_mt"),
    reference: str = DEFAULT_REFERENCE,
    measure: str = "chars",
) -> pd.DataFrame:
    """Tidy per-testimony lengths: one row per (scenario, testimony, target language)
    with the reference and target lengths and their ratio. Drives the blog's length
    figure (a dot per testimony, mean per language) and its CSV; characters by default
    because word counts understate the agglutinative Nguni languages."""
    rows = []
    for scenario in load_scenarios():
        for testimony in scenario.testimonies:
            tr = testimony.translations
            if reference not in tr:
                continue
            ref_len = _length(tr[reference].text, measure)
            for target in targets:
                if target in tr and ref_len:
                    tgt_len = _length(tr[target].text, measure)
                    rows.append(
                        {
                            "scenario_id": scenario.scenario_id,
                            "testimony_id": testimony.testimony_id,
                            "reference": reference,
                            "target": target,
                            "measure": measure,
                            f"len_{reference}": ref_len,
                            "len_target": tgt_len,
                            "ratio": tgt_len / ref_len,
                        }
                    )
    return pd.DataFrame(rows)


def with_lengths(
    df: pd.DataFrame, measure: str = "chars", lut: dict | None = None
) -> pd.DataFrame:
    """Attach length of the chosen and the rejected testimony to each valid trial.

    `lut` overrides the fixture-derived length table (used by tests)."""
    if lut is None:
        lut = length_lookup(measure)
    df = df.copy()

    def look(scenario: str, tid: str, lang: str):
        return lut.get((scenario, tid, lang))

    df["len_first"] = [
        look(s, t, lang)
        for s, t, lang in zip(
            df["metadata_scenario_id"],
            df["metadata_first_testimony_id"],
            df["metadata_lang_first"],
        )
    ]
    df["len_second"] = [
        look(s, t, lang)
        for s, t, lang in zip(
            df["metadata_scenario_id"],
            df["metadata_second_testimony_id"],
            df["metadata_lang_second"],
        )
    ]
    chose_first = df["chosen_id"] == df["metadata_first_testimony_id"]
    df["len_chosen"] = np.where(chose_first, df["len_first"], df["len_second"])
    df["len_other"] = np.where(chose_first, df["len_second"], df["len_first"])
    return df


def length_bias_controls(
    df: pd.DataFrame,
    measure: str = "chars",
    n_boot: int = 2000,
    seed: int = 0,
    lut: dict | None = None,
) -> dict:
    """Language-free verbosity bias: within same-language controls (no language
    difference), how often does the model pick the *longer* testimony? 0.5 = no length
    bias; >0.5 = prefers longer, <0.5 = prefers shorter. CI excluding 0.5 = real bias."""
    ctrl = with_lengths(
        df[(df["metadata_condition"] == CONTROL) & df["valid"]], measure, lut
    )
    ctrl = ctrl[ctrl["len_first"].notna() & ctrl["len_second"].notna()]
    ctrl = ctrl[ctrl["len_first"] != ctrl["len_second"]]
    if ctrl.empty:
        return {"n": 0, "p_chose_longer": float("nan"), "ci95": (float("nan"),) * 2}
    chose_longer = (ctrl["len_chosen"] > ctrl["len_other"]).astype(float)
    by_scenario = {
        s: chose_longer[ctrl["metadata_scenario_id"] == s].to_numpy()
        for s in ctrl["metadata_scenario_id"].unique()
    }
    lo, hi = _cluster_bootstrap_ci(by_scenario, n_boot, seed)
    return {
        "measure": measure,
        "n": int(len(ctrl)),
        "p_chose_longer": float(chose_longer.mean()),
        "ci95": (lo, hi),
        "significant": not (lo <= 0.5 <= hi),
    }


def length_bias_table(
    df: pd.DataFrame, measure: str = "chars", n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Per-model `length_bias_controls`: one row per model with P(chose the longer
    same-language testimony) and its scenario-bootstrap CI. The prose companion to the
    length figure (the figure shows the stimuli, this shows the models)."""
    rows = []
    for model in sorted(df["model"].astype(str).unique()):
        b = length_bias_controls(df[df["model"] == model], measure, n_boot, seed)
        lo, hi = b["ci95"]
        rows.append(
            {
                "model": model,
                "measure": measure,
                "n_control_trials": b["n"],
                "p_chose_longer": b["p_chose_longer"],
                "ci_lo": lo,
                "ci_hi": hi,
                "significant": bool(b.get("significant", False)),
            }
        )
    return pd.DataFrame(rows)


def effect_vs_length(
    df: pd.DataFrame,
    target: str = "afr",
    reference: str = DEFAULT_REFERENCE,
    measure: str = "chars",
) -> dict:
    """Per-scenario: does the model prefer the target language more where the target text is
    relatively longer? Correlates each scenario's target-preference (cross trials) with its
    target-vs-reference length advantage. Positive r => length tracks the language effect."""
    cross = _pair_cross_trials(df, target, reference)
    lut = length_lookup(measure)
    rows = []
    if not cross.empty:
        cross = cross.assign(
            chose_target=(cross["lang_chosen"] == target).astype(float)
        )
        per_scenario_pref = cross.groupby("metadata_scenario_id")["chose_target"].mean()
        for scenario in load_scenarios():
            sid = scenario.scenario_id
            if sid not in per_scenario_pref.index:
                continue
            deltas = [
                lut[(sid, t.testimony_id, target)]
                - lut[(sid, t.testimony_id, reference)]
                for t in scenario.testimonies
                if (sid, t.testimony_id, target) in lut
                and (sid, t.testimony_id, reference) in lut
            ]
            if deltas:
                rows.append(
                    {
                        "scenario_id": sid,
                        "target_pref": float(per_scenario_pref[sid]),
                        "len_delta": float(np.mean(deltas)),
                    }
                )
    table = pd.DataFrame(rows)
    r = (
        float(np.corrcoef(table["len_delta"], table["target_pref"])[0, 1])
        if len(table) > 2
        else float("nan")
    )
    return {
        "measure": measure,
        "n_scenarios": int(len(table)),
        "pearson_r": r,
        "table": table,
    }
