"""Counterbalancing is the spine of the design: if expansion mislabels which testimony
appears first, position bias and content preference become inseparable. These tests pin
that the two position orders are generated and that the metadata describing them is
internally consistent."""

from inspect_ai.dataset import Sample

from ufakazi.experiment.trials import comparison_pairs, expand_scenario
from ufakazi.scenarios.loader import Scenario, Testimony, Translation


def _scenario(langs: tuple[str, ...] = ("en", "af")) -> Scenario:
    prov = {"en": "source", "af": "human", "af_mt": "machine", "zu_mt": "machine"}

    def testimony(tid: str) -> Testimony:
        return Testimony(
            testimony_id=tid,
            speaker=f"Witness {tid}",
            translations={
                lang: Translation(text=f"text-{tid}-{lang}", provenance=prov[lang])
                for lang in langs
            },
        )

    return Scenario(
        scenario_id="s1",
        domain="test",
        question="Which is more credible?",
        testimonies=[testimony("A"), testimony("B")],
    )


def _meta(sample: Sample) -> dict:
    assert sample.metadata is not None
    return sample.metadata


def _user_text(sample: Sample) -> str:
    content = sample.input[-1].content  # type: ignore[union-attr]
    assert isinstance(content, str)
    return content


def test_expansion_counterbalances_position():
    samples = expand_scenario(_scenario(), languages=("en",))
    orders = {_meta(s)["position_order"] for s in samples}
    assert orders == {"AB", "BA"}, "both position orders must be present exactly once"
    assert len(samples) == 2


def test_metadata_first_second_match_position_order():
    for sample in expand_scenario(_scenario(), languages=("en",)):
        meta = _meta(sample)
        assert (
            meta["position_order"]
            == meta["first_testimony_id"] + meta["second_testimony_id"]
        )
        assert meta["first_testimony_id"] != meta["second_testimony_id"]


def test_first_presented_text_matches_first_testimony():
    # The user prompt must show the first testimony's text before the second's, so the
    # position label the model sees agrees with the recorded position_order.
    for sample in expand_scenario(_scenario(), languages=("en",)):
        meta = _meta(sample)
        user_text = _user_text(sample)
        first_id, second_id = meta["first_testimony_id"], meta["second_testimony_id"]
        assert user_text.index(f"text-{first_id}-en") < user_text.index(
            f"text-{second_id}-en"
        )


def test_language_assignment_recorded_per_language():
    samples = expand_scenario(_scenario(), languages=("af",))
    assert all(_meta(s)["lang_first"] == "af" for s in samples)
    assert all(_meta(s)["prov_first"] == "human" for s in samples)


def test_star_design_is_reference_centered():
    # Default star for {en, af}: the en control + the en-af cross pair only. One
    # same-language assignment (en,en) and both cross directions, each x 2 orders.
    samples = expand_scenario(_scenario(), languages=("en", "af"))
    conditions = [_meta(s)["condition"] for s in samples]
    assert conditions.count("same_language_control") == 2  # (en,en) x 2 orders
    assert conditions.count("cross_language") == 4  # (en,af)+(af,en) x 2 orders
    # The control is the reference language, never the target.
    controls = [s for s in samples if _meta(s)["condition"] == "same_language_control"]
    assert all(_meta(s)["lang_first"] == "en" for s in controls)


def test_star_design_omits_non_reference_cross_pairs():
    # zu_mt is a target, not the reference: it appears only against en, never against af.
    sc = _scenario(("en", "af", "zu_mt"))
    cross = [
        s
        for s in expand_scenario(sc, languages=("en", "af", "zu_mt"))
        if _meta(s)["condition"] == "cross_language"
    ]
    pairs = {
        frozenset((_meta(s)["lang_first"], _meta(s)["lang_second"])) for s in cross
    }
    assert frozenset(("af", "zu_mt")) not in pairs
    assert frozenset(("en", "zu_mt")) in pairs and frozenset(("en", "af")) in pairs


def test_star_design_includes_human_machine_provenance_pair():
    # af and af_mt share a base language, so the human-vs-machine pair is rendered even
    # though af is not the reference.
    sc = _scenario(("en", "af", "af_mt"))
    cross = {
        frozenset((_meta(s)["lang_first"], _meta(s)["lang_second"]))
        for s in expand_scenario(sc, languages=("en", "af", "af_mt"))
        if _meta(s)["condition"] == "cross_language"
    }
    assert frozenset(("af", "af_mt")) in cross


def test_full_design_is_the_complete_crossing():
    # |L|^2 assignments x 2 position orders; half are same-language controls for |L|=2.
    samples = expand_scenario(_scenario(), languages=("en", "af"), design="full")
    assert len(samples) == 8
    conditions = [_meta(s)["condition"] for s in samples]
    assert conditions.count("same_language_control") == 4
    assert conditions.count("cross_language") == 4


def test_comparison_pairs_star_vs_full():
    langs = ("en", "af", "af_mt")
    star = comparison_pairs(langs, design="star")
    # en control, en-af, en-af_mt, plus the af/af_mt provenance sibling. No af-only or
    # af_mt-only same-language control.
    assert set(star) == {("en",), ("af", "en"), ("af_mt", "en"), ("af", "af_mt")}
    full = comparison_pairs(langs, design="full")
    assert len(full) == 6  # 3 same-language + C(3,2)=3 cross


def test_language_is_bound_to_content_not_position():
    # The bias measure depends on language tracking *content* (testimony id), not the slot
    # it is shown in. Across the cross trials, content A must appear in each language
    # equally; that symmetry is what cancels content bias in the analysis.
    cross = [
        s
        for s in expand_scenario(_scenario(), languages=("en", "af"))
        if _meta(s)["condition"] == "cross_language"
    ]
    a_langs = []
    for sample in cross:
        text = _user_text(sample)
        a_lang = "en" if "text-A-en" in text else "af"
        b_lang = "en" if "text-B-en" in text else "af"
        assert a_lang != b_lang, (
            "a cross trial must clothe the two contents differently"
        )
        a_langs.append(a_lang)
    assert a_langs.count("en") == a_langs.count("af") == 2


def test_same_language_control_has_matching_languages():
    controls = [
        s
        for s in expand_scenario(_scenario(), languages=("en", "af"))
        if _meta(s)["condition"] == "same_language_control"
    ]
    for sample in controls:
        meta = _meta(sample)
        assert meta["lang_first"] == meta["lang_second"]
