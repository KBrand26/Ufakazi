"""Expand scenarios into counterbalanced Inspect samples.

A trial is one Inspect `Sample`. The trial tuple
`(scenario, language_assignment, position_order, model, system_prompt)` is split: model
and system prompt are fixed per eval run, while scenario, language assignment, and
position order are encoded here, the latter two into `Sample.metadata` so the scorer and
analysis can recover them.

M1 is naive on purpose: a single language assigned to both testimonies (a same-language
control) with position counterbalanced by swapping which testimony appears first. The
cross-language assignments that actually test H1 slot into the same loop later.
"""

from __future__ import annotations

from itertools import permutations

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from ufakazi.experiment.prompt import SYSTEM_PROMPT, render_user_prompt
from ufakazi.scenarios.loader import Scenario


def expand_scenario(
    scenario: Scenario, languages: tuple[str, ...] = ("en",)
) -> list[Sample]:
    """One sample per (language, position order). M1 keeps both testimonies in one
    language, so each language yields the two position orders of a same-language control."""
    if len(scenario.testimonies) != 2:
        raise ValueError(
            f"{scenario.scenario_id}: expected exactly 2 testimonies, "
            f"got {len(scenario.testimonies)}"
        )

    samples: list[Sample] = []
    for language in languages:
        for first, second in permutations(scenario.testimonies):  # (A,B) then (B,A)
            first_tr = first.render(language)
            second_tr = second.render(language)
            user_prompt = render_user_prompt(
                scenario.question, first_tr.text, second_tr.text
            )
            position_order = f"{first.testimony_id}{second.testimony_id}"
            metadata = {
                "scenario_id": scenario.scenario_id,
                "condition": "same_language_control",
                "position_order": position_order,
                "first_testimony_id": first.testimony_id,
                "second_testimony_id": second.testimony_id,
                "lang_first": language,
                "lang_second": language,
                "prov_first": first_tr.provenance,
                "prov_second": second_tr.provenance,
            }
            samples.append(
                Sample(
                    id=f"{scenario.scenario_id}|{language}-{language}|{position_order}",
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
