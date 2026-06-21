"""Load synthetic scenario fixtures from disk.

Scenarios are version-controlled YAML data, never live generation. Each testimony
carries one entry per language under `translations`, tagged with `provenance`
(`source` | `human` | `machine`) so the central translation confound stays first-class.
Machine translations are just another inline language entry: code `afr_mt`, provenance
`machine`, added alongside `en` and `afr`. No special handling, the loader parses any code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import yaml

DATA_DIR = Path(__file__).parent / "data"


# Markers a not-yet-written translation carries in the YAML. Translations land
# incrementally, so a placeholder must never render into a real trial; the assignable
# language set (`Scenario.languages`) excludes them until real text replaces the marker.
_PLACEHOLDER_PREFIXES = ("TBD", "[TODO", "[TBD", "TODO")


@dataclass(frozen=True)
class Translation:
    """One language rendering of a testimony."""

    text: str
    provenance: str  # "source" | "human" | "machine"

    @property
    def is_placeholder(self) -> bool:
        """True for an empty or stub (`TBD` / `[TODO ...]`) text awaiting a real translation."""
        marker = self.text.strip().upper()
        return not marker or marker.startswith(_PLACEHOLDER_PREFIXES)


@dataclass(frozen=True)
class Testimony:
    __test__: ClassVar[bool] = False  # not a pytest class despite the "Test..." prefix

    testimony_id: str
    speaker: str
    translations: dict[str, Translation]  # language code -> rendering

    def render(self, language: str) -> Translation:
        """Return the rendering in `language`, raising if it is not present."""
        try:
            return self.translations[language]
        except KeyError as exc:
            raise KeyError(
                f"testimony {self.testimony_id!r} has no {language!r} translation "
                f"(have: {sorted(self.translations)})"
            ) from exc


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    domain: str
    question: str
    testimonies: list[Testimony]

    def languages(self) -> set[str]:
        """Languages assignable for every testimony: present and non-placeholder on each."""
        return set.intersection(
            *(
                {lang for lang, tr in t.translations.items() if not tr.is_placeholder}
                for t in self.testimonies
            )
        )


def _parse_scenario(raw: dict) -> Scenario:
    testimonies = [
        Testimony(
            testimony_id=t["testimony_id"],
            speaker=t.get("speaker", t["testimony_id"]),
            translations={
                lang: Translation(text=tr["text"].strip(), provenance=tr["provenance"])
                for lang, tr in t["translations"].items()
            },
        )
        for t in raw["testimonies"]
    ]
    return Scenario(
        scenario_id=raw["scenario_id"],
        domain=raw.get("domain", "unspecified"),
        question=raw["question"].strip(),
        testimonies=testimonies,
    )


def load_scenario(path: Path) -> Scenario:
    with open(path, encoding="utf-8") as fh:
        return _parse_scenario(yaml.safe_load(fh))


def load_scenarios(data_dir: Path = DATA_DIR) -> list[Scenario]:
    """Load every `*.yaml` fixture in `data_dir`, sorted by filename for stable order."""
    return [load_scenario(p) for p in sorted(data_dir.glob("*.yaml"))]
