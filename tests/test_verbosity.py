"""The verbosity check is the rebuttal to "isn't it just length?", so its core
computation must be right: within same-language controls, does the model pick the longer
testimony? These tests pin the chose-longer logic against an injected length table, with
no dependency on the real fixtures."""

import pandas as pd

from ufakazi.analysis.verbosity import length_bias_controls, with_lengths

# Injected length table: (scenario, testimony, language) -> length.
# In every scenario, content B is the longer testimony.
LUT = {
    ("s0", "A", "en"): 100,
    ("s0", "B", "en"): 200,
    ("s1", "A", "en"): 100,
    ("s1", "B", "en"): 200,
}


def _control_row(scenario: str, first: str, second: str, chosen: str):
    return {
        "metadata_condition": "same_language_control",
        "metadata_scenario_id": scenario,
        "metadata_first_testimony_id": first,
        "metadata_second_testimony_id": second,
        "metadata_lang_first": "en",
        "metadata_lang_second": "en",
        "chosen_id": chosen,
        "valid": True,
    }


def test_with_lengths_resolves_chosen_and_other():
    df = pd.DataFrame([_control_row("s0", "A", "B", "B")])
    out = with_lengths(df, lut=LUT)
    assert out["len_chosen"].iloc[0] == 200  # chose B, the longer
    assert out["len_other"].iloc[0] == 100


def test_always_choosing_longer_gives_rate_one():
    # Model always picks content B (the longer one), in both position orders.
    rows = [_control_row(s, "A", "B", "B") for s in ("s0", "s1")]
    rows += [_control_row(s, "B", "A", "B") for s in ("s0", "s1")]
    result = length_bias_controls(pd.DataFrame(rows), lut=LUT, seed=1)
    assert result["p_chose_longer"] == 1.0
    assert result["significant"]


def test_no_length_bias_sits_at_half():
    # Half the trials pick the longer (B), half the shorter (A): no length preference.
    rows = [
        _control_row("s0", "A", "B", "B"),  # longer
        _control_row("s0", "A", "B", "A"),  # shorter
        _control_row("s1", "A", "B", "B"),  # longer
        _control_row("s1", "A", "B", "A"),  # shorter
    ]
    result = length_bias_controls(pd.DataFrame(rows), lut=LUT, seed=1)
    assert result["p_chose_longer"] == 0.5
    assert not result["significant"]
