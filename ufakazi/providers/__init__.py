"""Model configuration.

Inspect's own model layer (`inspect_ai.model` plus its 20+ provider backends) supersedes
the hand-rolled provider adapters originally planned for this package, including the
per-provider logprob handling. What remains here is Ufakazi's model-selection defaults
and the generation config.
"""

from __future__ import annotations

from inspect_ai.model import GenerateConfig

MOCK_MODEL = "mockllm/model"
DEFAULT_MODEL = "openai/gpt-4o-mini"  # used when a real API key is available


def generation_config(**overrides) -> GenerateConfig:
    """Deterministic generation requesting choice-token logprobs.

    `logprobs` is honored by providers that support it (OpenAI) and ignored by those that
    do not (Anthropic, the mock); the forced binary choice is the universal fallback.
    """
    base: dict = dict(temperature=0.0, max_tokens=200, logprobs=True, top_logprobs=1)
    base.update(overrides)
    return GenerateConfig(**base)
