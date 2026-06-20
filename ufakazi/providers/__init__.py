"""Model configuration: the static registry of selectable models and generation config.

Inspect's own model layer (`inspect_ai.model` plus its provider backends) supersedes the
hand-rolled provider adapters originally planned here, including per-provider logprob
handling. We reach providers through **OpenRouter** by default (`openrouter/...`, one API
key, one credit pool); native `openai/` / `anthropic/` strings still work as a fallback.

This module is intentionally pure config (no model construction, no experiment imports):
the registry is the single source of truth for the interactive picker and for
config-driven multi-model runs. Resolving a spec to a live model object (including the
keyless mock) lives in `experiment.models`.
"""

from __future__ import annotations

from dataclasses import dataclass

from inspect_ai.model import GenerateConfig

MOCK_MODEL = "mockllm/model"


@dataclass(frozen=True)
class ModelOption:
    """One selectable model: a short `key`, its Inspect model string, and display text."""

    key: str
    model: str
    label: str
    note: str = ""


# Ordered; the default real model is `DEFAULT_KEY`. OpenRouter slugs other than
# gpt-4o-mini are best-effort and worth confirming with the `probe` command before a run.
MODEL_REGISTRY: tuple[ModelOption, ...] = (
    ModelOption(
        "mock",
        MOCK_MODEL,
        "Mock (keyless)",
        "Deterministic position-biased responder. No API key or spend.",
    ),
    ModelOption(
        "gpt-4o-mini",
        "openrouter/openai/gpt-4o-mini",
        "GPT-4o mini - OpenRouter",
        "Cheap; returns choice-token logprobs. Default real model.",
    ),
    ModelOption(
        "claude-haiku",
        "openrouter/anthropic/claude-3.5-haiku",
        "Claude 3.5 Haiku - OpenRouter",
        "No logprobs; exercises the binary-choice fallback.",
    ),
    ModelOption(
        "llama-free",
        "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        "Llama 3.3 70B - OpenRouter (free)",
        "Zero-cost smoke option; logprobs not guaranteed.",
    ),
)

DEFAULT_KEY = "gpt-4o-mini"
DEFAULT_MODEL = next(o.model for o in MODEL_REGISTRY if o.key == DEFAULT_KEY)


def option_for(key: str) -> ModelOption | None:
    """Look up a registry option by its short key, or None if not registered."""
    return next((o for o in MODEL_REGISTRY if o.key == key), None)


def generation_config(**overrides) -> GenerateConfig:
    """Deterministic generation requesting choice-token logprobs.

    `logprobs` is honored by providers that support it (OpenAI family) and ignored by
    those that do not (Anthropic, the mock); the forced binary choice is the universal
    fallback. Under unpinned OpenRouter routing, whether logprobs return can also vary by
    the backend a request lands on, hence the `probe` command.
    """
    base: dict = dict(temperature=0.0, max_tokens=200, logprobs=True, top_logprobs=5)
    base.update(overrides)
    return GenerateConfig(**base)
