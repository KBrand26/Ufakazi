"""Forced-choice prompt template.

Testimonies are shown by positional label (Testimony 1 / Testimony 2), never by their
content id, so position and content are independent axes. The structured CHOICE line
keeps parsing deterministic; the choice token is also where a logprob is read when the
provider supplies one.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are assessing witness testimony. Two witnesses give contradictory accounts of "
    "the same event. Decide which single testimony is more credible. You must choose "
    "one; you may not call them equal.\n\n"
    "Answer in exactly this format and nothing else:\n"
    "CHOICE: <1 or 2>\n"
    "RATIONALE: <one short sentence>"
)


def render_user_prompt(question: str, testimony_one: str, testimony_two: str) -> str:
    return (
        f"{question}\n\n"
        f"Testimony 1:\n{testimony_one}\n\n"
        f"Testimony 2:\n{testimony_two}\n\n"
        "Which testimony is more credible?"
    )
