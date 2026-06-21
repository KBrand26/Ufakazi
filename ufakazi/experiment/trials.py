"""Expand scenarios into counterbalanced Inspect samples.

A trial is one Inspect `Sample`. The trial tuple
`(scenario, language_assignment, position_order, model, system_prompt)` is split: model
and system prompt are fixed per eval run, while scenario, language assignment, and
position order are encoded here, the latter two into `Sample.metadata` so the scorer and
analysis can recover them.

Languages are clothed onto the two *contents* (testimony identities). Which language
pairs we render is the **design**:

- **star** (default): every other language against a single `reference` (English),
  plus the same-language reference control, plus any human-vs-machine provenance pair
  (a `{x}_mt` machine code whose `{x}` human base is also present, e.g. `afr`/`afr_mt`).
  Every emitted pair maps to a hypothesis; the cross terms we do not ask about
  (e.g. zul_mt x xho_mt) are not spent.
- **full**: the complete `|L|^2` crossing, every language against every other.

For each *cross* pair `{x, y}` both content->language assignments are emitted (content A
in x / B in y, and the swap) and crossed with both position orders; that symmetry cancels
content and position bias structurally, isolating the language main effect (see
`analysis/`). A same-language pair `{x}` yields one assignment over both orders and is the
position/content control. Replication across the provider's residual nondeterminism is
handled by Inspect `epochs`, not here.
"""

from __future__ import annotations

import logging
from itertools import permutations
from typing import Literal

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from ufakazi.experiment.prompt import SYSTEM_PROMPT, render_user_prompt
from ufakazi.scenarios.loader import Scenario

logger = logging.getLogger(__name__)

SAME_LANGUAGE_CONTROL = "same_language_control"
CROSS_LANGUAGE = "cross_language"

Design = Literal["star", "full"]
DEFAULT_DESIGN: Design = "star"
DEFAULT_REFERENCE = "en"
MT_SUFFIX = "_mt"


def _resolve_reference(langs: list[str], reference: str | None) -> str:
    """The hub language for the star design. An explicit reference must be present;
    otherwise prefer English, falling back to the first language so single-language
    expansions (and en-less runs) still have a well-defined hub."""
    if reference is not None:
        if reference not in langs:
            raise ValueError(f"reference {reference!r} not in languages {langs}")
        return reference
    return DEFAULT_REFERENCE if DEFAULT_REFERENCE in langs else langs[0]


def _provenance_sibling_pairs(langs: list[str]) -> set[frozenset[str]]:
    """Human-vs-machine pairs implied by the `_mt` naming convention: a machine code
    `{x}_mt` whose human base `{x}` is also present (e.g. `afr`/`afr_mt`). Machine-only
    languages (no human base in the run) contribute nothing, which is correct."""
    return {
        frozenset((code[: -len(MT_SUFFIX)], code))
        for code in langs
        if code.endswith(MT_SUFFIX) and code[: -len(MT_SUFFIX)] in langs
    }


def comparison_pairs(
    languages: tuple[str, ...],
    *,
    reference: str | None = None,
    design: str = DEFAULT_DESIGN,
) -> list[tuple[str, ...]]:
    """The unordered language pairs to render, as sorted tuples (length 1 = same-language
    control). `full` is the complete crossing; `star` is reference-centered plus
    provenance siblings. Order is stable for reproducible sample ids."""
    langs = sorted(set(languages))
    if not langs:
        return []
    if design == "full":
        pairs = {frozenset((a, b)) for a in langs for b in langs}
    elif design == "star":
        ref = _resolve_reference(langs, reference)
        pairs = {frozenset((ref,))}
        pairs |= {frozenset((ref, t)) for t in langs if t != ref}
        pairs |= _provenance_sibling_pairs(langs)
    else:
        raise ValueError(f"unknown design {design!r} (expected 'star' or 'full')")
    return sorted((tuple(sorted(p)) for p in pairs), key=lambda p: (len(p), p))


def _ordered_assignments(pairs: list[tuple[str, ...]]) -> list[tuple[str, str]]:
    """Expand each unordered pair into ordered (content_a language, content_b language)
    assignments: a same-language pair gives one, a cross pair gives both directions so
    each content carries each language equally."""
    ordered: list[tuple[str, str]] = []
    for pair in pairs:
        if len(pair) == 1:
            ordered.append((pair[0], pair[0]))
        else:
            a, b = pair
            ordered.append((a, b))
            ordered.append((b, a))
    return ordered


def expand_scenario(
    scenario: Scenario,
    languages: tuple[str, ...] = ("en",),
    *,
    reference: str | None = None,
    design: str = DEFAULT_DESIGN,
) -> list[Sample]:
    """One sample per (language assignment, position order) for the chosen `design`.

    The language assignment maps a language to each *content* (testimony identity); the
    position order decides which content is presented first.
    """
    if len(scenario.testimonies) != 2:
        raise ValueError(
            f"{scenario.scenario_id}: expected exactly 2 testimonies, "
            f"got {len(scenario.testimonies)}"
        )

    pairs = comparison_pairs(languages, reference=reference, design=design)
    content_a, content_b = scenario.testimonies

    samples: list[Sample] = []
    for lang_a, lang_b in _ordered_assignments(pairs):
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
    scenarios: list[Scenario],
    languages: tuple[str, ...] = ("en",),
    *,
    reference: str | None = None,
    design: str = DEFAULT_DESIGN,
) -> list[Sample]:
    """Expand the scenarios that carry every requested language into counterbalanced samples.

    A scenario only counts a language once every testimony carries a real (non-placeholder)
    translation for it, so half-written scenarios (new ones start `en`/`afr` only, with the
    rest stubbed `TBD`) are dropped with a warning rather than crashing the run or rendering
    a placeholder into a trial. Raises if none qualify.
    """
    want = set(languages)
    supported, skipped = [], []
    for scenario in scenarios:
        if want <= scenario.languages():
            supported.append(scenario)
        else:
            skipped.append(scenario.scenario_id)
    if skipped:
        logger.warning(
            "Skipping %d/%d scenario(s) missing requested languages %s: %s",
            len(skipped),
            len(scenarios),
            sorted(want),
            ", ".join(skipped),
        )
    if not supported:
        raise ValueError(
            f"No scenarios carry all requested languages {sorted(want)}; "
            f"checked {len(scenarios)} scenario(s)."
        )
    return [
        sample
        for scenario in supported
        for sample in expand_scenario(
            scenario, languages, reference=reference, design=design
        )
    ]
