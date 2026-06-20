"""Expand scenarios into counterbalanced Inspect samples.

A trial is one Inspect `Sample`. The trial tuple
`(scenario, language_assignment, position_order, model, system_prompt)` is split: model
and system prompt are fixed per eval run, while scenario, language assignment, and
position order are encoded here, the latter two into `Sample.metadata` so the scorer and
analysis can recover them.

Languages are treated as a **set**: for a two-testimony scenario and a language set L we
emit every ordered assignment of languages to the two *contents* (|L|^2 of them) crossed
with both position orders. When the two contents share a language the trial is a
`same_language_control` (the position/content baseline); when they differ it is a
`cross_language` trial, and aggregating the cross trials symmetrically cancels content
bias to isolate the language main effect (see `analysis/`). Replication across the
provider's residual nondeterminism is handled by Inspect `epochs`, not here.
"""

from __future__ import annotations

from itertools import permutations, product

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from ufakazi.experiment.prompt import SYSTEM_PROMPT, render_user_prompt
from ufakazi.scenarios.loader import Scenario

SAME_LANGUAGE_CONTROL = "same_language_control"
CROSS_LANGUAGE = "cross_language"


def expand_scenario(
    scenario: Scenario, languages: tuple[str, ...] = ("en",)
) -> list[Sample]:
    """One sample per (language assignment, position order).

    The language assignment maps a language to each *content* (testimony identity), the
    position order decides which content is presented first. With |L| languages that is
    |L|^2 assignments * 2 orders samples, half same-language controls (when |L|>1).
    """
    if len(scenario.testimonies) != 2:
        raise ValueError(
            f"{scenario.scenario_id}: expected exactly 2 testimonies, "
            f"got {len(scenario.testimonies)}"
        )

    langs = sorted(set(languages))
    content_a, content_b = scenario.testimonies

    samples: list[Sample] = []
    for lang_a, lang_b in product(langs, langs):
        by_content = {content_a.testimony_id: lang_a, content_b.testimony_id: lang_b}
        condition = SAME_LANGUAGE_CONTROL if lang_a == lang_b else CROSS_LANGUAGE
        for first, second in permutations(scenario.testimonies):  # (A,B) then (B,A)
            lang_first = by_content[first.testimony_id]
            lang_second = by_content[second.testimony_id]
            first_tr = first.render(lang_first)
            second_tr = second.render(lang_second)
            user_prompt = render_user_prompt(
                scenario.question, first_tr.text, second_tr.text
            )
            position_order = f"{first.testimony_id}{second.testimony_id}"
            metadata = {
                "scenario_id": scenario.scenario_id,
                "condition": condition,
                "position_order": position_order,
                "first_testimony_id": first.testimony_id,
                "second_testimony_id": second.testimony_id,
                "lang_first": lang_first,
                "lang_second": lang_second,
                "prov_first": first_tr.provenance,
                "prov_second": second_tr.provenance,
            }
            samples.append(
                Sample(
                    id=f"{scenario.scenario_id}|{lang_a}_{lang_b}|{position_order}",
                    input=[
                        ChatMessageSystem(content=SYSTEM_PROMPT),
                        ChatMessageUser(content=user_prompt),
                    ],
                    metadata=metadata,
                )
            )
    return samples


def build_dataset(
    scenarios: list[Scenario], languages: tuple[str, ...] = ("en",)
) -> list[Sample]:
    return [s for scenario in scenarios for s in expand_scenario(scenario, languages)]
