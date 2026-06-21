"""Resolve a CLI/model spec into something Inspect's `eval()` accepts.

Kept separate from `providers` (pure static config) because building the keyless mock
needs the experiment-specific position-biased responder. This is the one place that knows
how every spec - a bare key, the `default` alias, the mock, or a raw Inspect string -
turns into a live model.
"""

from __future__ import annotations

from inspect_ai.model import GenerateConfig, Model, get_model

from ufakazi.experiment.mock import position_biased_responder
from ufakazi.providers import DEFAULT_MODEL, MOCK_MODEL, generation_config, option_for


def mock_model() -> Model:
    """The keyless mock wired with the deterministic position-biased responder."""
    return get_model(MOCK_MODEL, custom_outputs=position_biased_responder(1))


def resolve_model(spec: str | None) -> str | Model:
    """Map a spec to a value `eval(model=...)` accepts.

    - `None`, `"mock"`, or the mock string -> the keyless mock `Model`.
    - `"default"` -> the default real model string.
    - a registry key -> a `Model` built with that option's per-model config and provider
      pin baked in (mock key -> mock `Model`), so reasoning/logprob/routing settings travel
      with the model. Raw strings fall back to the default config at `eval()` time.
    - anything else -> treated as a raw Inspect model string (e.g. `openrouter/...`).
    """
    if spec is None or spec == "mock" or spec == MOCK_MODEL:
        return mock_model()
    if spec == "default":
        return DEFAULT_MODEL
    option = option_for(spec)
    if option is not None:
        if option.model == MOCK_MODEL:
            return mock_model()
        return get_model(
            option.model, config=option.generate_config(), **option.model_args()
        )
    return spec


def task_config_for(spec: str | None) -> GenerateConfig:
    """The `GenerateConfig` the task should run under for this spec.

    Inspect makes the task ("active") config override the model's own config, so the task
    must carry the *same* per-model config the model was built with - otherwise a global
    `logprobs=True` would clobber a reasoning model's `logprobs=False`. Mock / `default` /
    raw strings fall back to the logprob-requesting default.
    """
    option = (
        option_for(spec) if spec not in (None, "mock", MOCK_MODEL, "default") else None
    )
    if option is not None and option.model != MOCK_MODEL:
        return option.generate_config()
    return generation_config()


def build_model(spec: str | None) -> Model:
    """Like `resolve_model` but always returns a live `Model` carrying our generation
    config (logprobs requested). Used by the probe, which calls `generate` directly."""
    resolved = resolve_model(spec)
    if isinstance(resolved, Model):
        return resolved
    return get_model(resolved, config=generation_config())
