"""Deterministic mock responder for keyless, costless flow tests.

This is not a model. It is a fixed responder wired into Inspect's `mockllm` provider so
the whole M1 pipeline runs end-to-end with no API key and no spend. It always picks a
fixed *position*, simulating a pure position bias, so the analysis has a known signal to
recover: with position counterbalanced, this should surface as a ~100% first-position
rate and a ~50/50 content split, exactly what the same-language control is meant to show.
"""

from __future__ import annotations

from collections.abc import Callable

from inspect_ai.model import ModelOutput

from ufakazi.providers import MOCK_MODEL


def position_biased_responder(bias_to_position: int = 1) -> Callable:
    """Build a mockllm `custom_outputs` callable that always answers `bias_to_position`."""

    def respond(input, tools, tool_choice, config) -> ModelOutput:  # noqa: A002
        return ModelOutput.from_content(
            model=MOCK_MODEL,
            content=(
                f"CHOICE: {bias_to_position}\n"
                "RATIONALE: (mock) deterministically selects a fixed position."
            ),
        )

    return respond
