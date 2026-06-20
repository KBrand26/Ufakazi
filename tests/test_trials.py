"""Counterbalancing is the spine of the design: if expansion mislabels which testimony
appears first, position bias and content preference become inseparable. These tests pin
that the two position orders are generated and that the metadata describing them is
internally consistent."""

from inspect_ai.dataset import Sample

from ufakazi.experiment.trials import expand_scenario
from ufakazi.scenarios.loader import Scenario, Testimony, Translation


def _scenario() -> Scenario:
    def testimony(tid: str) -> Testimony:
        return Testimony(
            testimony_id=tid,
            speaker=f"Witness {tid}",
            translations={
                "en": Translation(text=f"text-{tid}-en", provenance="source"),
                "af": Translation(text=f"text-{tid}-af", provenance="human"),
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
