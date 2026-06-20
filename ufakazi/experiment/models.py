"""Resolve a CLI/model spec into something Inspect's `eval()` accepts.

Kept separate from `providers` (pure static config) because building the keyless mock
needs the experiment-specific position-biased responder. This is the one place that knows
how every spec - a bare key, the `default` alias, the mock, or a raw Inspect string -
turns into a live model.
"""

from __future__ import annotations

from inspect_ai.model import Model, get_model

from ufakazi.experiment.mock import position_biased_responder
from ufakazi.providers import DEFAULT_MODEL, MOCK_MODEL, generation_config, option_for


def mock_model() -> Model:
    """The keyless mock wired with the deterministic position-biased responder."""
    return get_model(MOCK_MODEL, custom_outputs=position_biased_responder(1))


def resolve_model(spec: str | None) -> str | Model:
    """Map a spec to a value `eval(model=...)` accepts.

    - `None`, `"mock"`, or the mock string -> the keyless mock `Model`.
    - `"default"` -> the default real model string.
    - a registry key (e.g. `"gpt-4o-mini"`) -> its model string (mock key -> mock `Model`).
    - anything else -> treated as a raw Inspect model string (e.g. `openrouter/...`).
    """
    if spec is None or spec == "mock" or spec == MOCK_MODEL:
        return mock_model()
    if spec == "default":
        return DEFAULT_MODEL
    option = option_for(spec)
    if option is not None:
        return mock_model() if option.model == MOCK_MODEL else option.model
    return spec


def build_model(spec: str | None) -> Model:
    """Like `resolve_model` but always returns a live `Model` carrying our generation
    config (logprobs requested). Used by the probe, which calls `generate` directly."""
    resolved = resolve_model(spec)
    if isinstance(resolved, Model):
        return resolved
    return get_model(resolved, config=generation_config())
