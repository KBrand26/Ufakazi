"""Parsing the forced choice and mapping position back to content is where a wrong result
would silently hide. Test the parser against realistic and malformed completions, and the
position->content join under both counterbalanced orders."""

import pytest

from ufakazi.experiment.scoring import parse_choice, parse_rationale, resolve_choice


@pytest.mark.parametrize(
    "text, expected",
    [
        ("CHOICE: 1\nRATIONALE: clearer vantage", 1),
        ("CHOICE: 2", 2),
        ("choice: 2  ", 2),  # case-insensitive, trailing space
        ("Sure!\nCHOICE: 1\nRATIONALE: ...", 1),  # leading chatter
        ("I think they are equal", None),  # refused the format
        ("CHOICE: 3", None),  # out of range
        ("", None),
        (None, None),
    ],
)
def test_parse_choice(text, expected):
    assert parse_choice(text) == expected


def test_resolve_choice_separates_position_from_content():
    # AB order: position 1 is testimony A; BA order: position 1 is testimony B.
    assert resolve_choice(1, "A", "B") == "A"
    assert resolve_choice(2, "A", "B") == "B"
    assert resolve_choice(1, "B", "A") == "B"
    assert resolve_choice(2, "B", "A") == "A"
    assert resolve_choice(None, "A", "B") is None


def test_parse_rationale():
    assert parse_rationale("CHOICE: 1\nRATIONALE: clearer vantage") == "clearer vantage"
    assert parse_rationale("CHOICE: 1") is None
