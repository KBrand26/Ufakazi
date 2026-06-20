"""Record-only scorer.

There is no correct answer here: we record which testimony the model chose, not whether
it was right. The scorer parses the forced CHOICE, maps the chosen *position* back to the
chosen *content* via sample metadata, and records both plus the choice logprob when the
provider supplies one. Preference rates and the position baseline are computed downstream
in `ufakazi.analysis`; the two metrics here are just quick run-level sanity readouts.
"""

from __future__ import annotations

import re

from inspect_ai.scorer import SampleScore, Score, Target, metric, scorer
from inspect_ai.solver import TaskState

_CHOICE_RE = re.compile(r"CHOICE:\s*([12])", re.IGNORECASE)
_RATIONALE_RE = re.compile(r"RATIONALE:\s*(.+)", re.IGNORECASE)

PARSE_ERROR = "PARSE_ERROR"


def parse_choice(text: str | None) -> int | None:
    """Extract the chosen position (1 or 2) from a model completion, else None."""
    match = _CHOICE_RE.search(text or "")
    return int(match.group(1)) if match else None


def parse_rationale(text: str | None) -> str | None:
    match = _RATIONALE_RE.search(text or "")
    return match.group(1).strip() if match else None


def resolve_choice(position: int | None, first_id: str, second_id: str) -> str | None:
    """Map a chosen position back to the content (testimony) id it landed on.

    This is the join that separates position bias from content/language preference: the
    same position means a different testimony depending on the counterbalanced order.
    """
    if position == 1:
        return first_id
    if position == 2:
        return second_id
    return None


def _choice_logprob(state: TaskState, position: int | None) -> float | None:
    """Logprob of the chosen digit token (1/2) on the CHOICE line.

    We want the model's confidence in its *choice*, not the first/format token: under the
    forced template the first token (`CHOICE`) is trivially near-certain (logprob ~0) and
    carries no signal. Scan tokens up to the first newline (the CHOICE line, before any
    RATIONALE digits) and return the logprob of the token equal to the chosen digit.
    Returns None when the provider supplied no logprobs (Anthropic, the mock)."""
    choices = state.output.choices
    if position is None or not (choices and choices[0].logprobs):
        return None
    target = str(position)
    for token in choices[0].logprobs.content or []:
        if "\n" in token.token:
            break
        if token.token.strip() == target:
            return token.logprob
    return None


@metric
def chose_first_position_rate():
    """Fraction of valid trials that picked the first-presented testimony (position
    baseline: ~0.5 means no position bias)."""

    def compute(scores: list[SampleScore]) -> float:
        positions = [(s.score.metadata or {}).get("chosen_position") for s in scores]
        valid = [p for p in positions if p is not None]
        return sum(1 for p in valid if p == 1) / len(valid) if valid else float("nan")

    return compute


@metric
def parse_failure_rate():
    """Fraction of trials whose CHOICE line could not be parsed."""

    def compute(scores: list[SampleScore]) -> float:
        if not scores:
            return float("nan")
        failed = sum(
            1 for s in scores if (s.score.metadata or {}).get("chosen_position") is None
        )
        return failed / len(scores)

    return compute


@scorer(metrics=[chose_first_position_rate(), parse_failure_rate()])
def record_choice():
    async def score(state: TaskState, target: Target) -> Score:
        text = state.output.completion
        position = parse_choice(text)
        meta = state.metadata
        chosen_id = resolve_choice(
            position, meta["first_testimony_id"], meta["second_testimony_id"]
        )
        return Score(
            value=chosen_id if chosen_id is not None else PARSE_ERROR,
            answer=text,
            explanation=parse_rationale(text),
            metadata={
                "chosen_position": position,
                "chosen_testimony_id": chosen_id,
                "choice_logprob": _choice_logprob(state, position),
            },
        )

    return score
