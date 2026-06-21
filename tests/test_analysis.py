"""The analysis is where a wrong result hides. The load-and-flatten is glue, but the
content-bias cancellation is the whole claim: cross-language aggregation must report no
language effect for a language-blind model even when one testimony's content is always
preferred, and must report a real effect when the model actually favors a language. These
tests pin both, plus the continuous logprob measure, on synthetic frames."""

import math

import pandas as pd

from ufakazi.analysis.load import (
    _continuous_target_prob,
    filter_latest_per_model,
    language_effect_table,
    language_preference,
    per_scenario_language_effect,
)


def _cross_row(scenario: str, lang_a: str, lang_b: str, chosen_id: str, logprob=None):
    """One cross-language trial row in the shape the analysis consumes."""
    return {
        "metadata_condition": "cross_language",
        "metadata_scenario_id": scenario,
        "lang_A": lang_a,
        "lang_B": lang_b,
        "chosen_id": chosen_id,
        "lang_chosen": lang_a if chosen_id == "A" else lang_b,
        "valid": True,
        "choice_logprob": logprob,
    }


def _blind_always_A(scenarios: list[str]) -> pd.DataFrame:
    """Language-blind model with maximal content bias: always picks content A. Each
    scenario contributes both language assignments (A=afr,B=en) and (A=en,B=afr)."""
    rows = []
    for s in scenarios:
        rows.append(_cross_row(s, "afr", "en", "A"))  # picks A, which is afr
        rows.append(_cross_row(s, "en", "afr", "A"))  # picks A, which is en
    return pd.DataFrame(rows)


def _afr_favoring(scenarios: list[str]) -> pd.DataFrame:
    """Model that always picks whichever testimony is written in Afrikaans."""
    rows = []
    for s in scenarios:
        rows.append(_cross_row(s, "afr", "en", "A"))  # afr is on A -> pick A
        rows.append(_cross_row(s, "en", "afr", "B"))  # afr is on B -> pick B
    return pd.DataFrame(rows)


SCENARIOS = [f"s{i}" for i in range(6)]


def test_content_bias_cancels_for_language_blind_model():
    # Always-pick-A is a huge content bias, but A is afr in half the trials and en in the
    # other half, so the language preference must come out at 0.5 and not significant.
    result = language_preference(_blind_always_A(SCENARIOS), target="afr", seed=1)
    assert result["p_prefer_target"] == 0.5
    assert not result["significant"]


def test_real_language_preference_is_detected():
    result = language_preference(_afr_favoring(SCENARIOS), target="afr", seed=1)
    assert result["p_prefer_target"] == 1.0
    assert result["significant"]
    lo, _hi = result["ci95"]
    assert lo > 0.5  # CI excludes the no-effect null


def test_per_scenario_effect_sign_and_magnitude():
    # afr-favoring model: when A is afr it always picks A (1.0), when A is en it never does
    # (0.0), so the per-scenario language effect is +1.0 for every scenario.
    effects = per_scenario_language_effect(_afr_favoring(SCENARIOS), target="afr")
    assert len(effects) == len(SCENARIOS)
    assert (effects["language_effect"] == 1.0).all()


def test_continuous_measure_uses_chosen_token_probability():
    # Chose the afr testimony with logprob ln(0.8): P(afr) = 0.8. Chose the en testimony
    # with logprob ln(0.9): it put 0.9 on en, so 0.1 on afr. Mean = 0.45.
    rows = [
        _cross_row("s0", "afr", "en", "A", logprob=math.log(0.8)),  # chose afr
        _cross_row("s0", "en", "afr", "A", logprob=math.log(0.9)),  # chose en
    ]
    cross = pd.DataFrame(rows)
    assert _continuous_target_prob(cross, "afr") == 0.45


def test_language_effect_table_does_not_conflate_models():
    # The whole point of model-awareness: an afr-favoring model and a language-blind one
    # in the same frame must yield separate, correct per-model rows, not a pooled average.
    afr = _afr_favoring(SCENARIOS).assign(model="favors_afr")
    blind = _blind_always_A(SCENARIOS).assign(model="language_blind")
    table = language_effect_table(pd.concat([afr, blind], ignore_index=True), seed=1)
    assert set(table["model"]) == {"favors_afr", "language_blind"}
    favors = table[table["model"] == "favors_afr"].iloc[0]
    neutral = table[table["model"] == "language_blind"].iloc[0]
    assert favors["p_prefer_target"] == 1.0 and favors["significant"]
    assert neutral["p_prefer_target"] == 0.5 and not neutral["significant"]


def test_filter_latest_per_model_keeps_each_models_most_recent_eval():
    # Model A has two evals; the newer (lexicographically-later log) wins. Model B's single
    # eval survives. A naive global "latest eval" would have dropped B entirely.
    df = pd.DataFrame(
        [
            {"model": "A", "eval_id": "a_old", "log": "2026-01-01_t.eval"},
            {"model": "A", "eval_id": "a_new", "log": "2026-02-01_t.eval"},
            {"model": "B", "eval_id": "b_one", "log": "2026-01-15_t.eval"},
        ]
    )
    kept = set(filter_latest_per_model(df)["eval_id"])
    assert kept == {"a_new", "b_one"}


def test_per_scenario_effect_empty_when_no_paired_assignments():
    # No scenario has both an A=target and an A=reference trial, so no paired difference
    # can be formed. Must return an empty, correctly-columned frame (not raise).
    rows = [_cross_row("s0", "afr", "en", "A"), _cross_row("s1", "afr", "en", "B")]
    effects = per_scenario_language_effect(pd.DataFrame(rows), target="afr")
    assert effects.empty
    assert list(effects.columns) == ["scenario_id", "language_effect", "n_trials"]
