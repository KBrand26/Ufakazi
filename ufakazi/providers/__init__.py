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


# Token headroom: reasoning models need room for the thinking trace plus the short
# answer; non-reasoning models emit only the forced choice + a one-line rationale.
REASONING_MAX_TOKENS = 3000
CHOICE_MAX_TOKENS = 200


@dataclass(frozen=True)
class ModelOption:
    """One selectable model and how to call it.

    Beyond identity (`key`, `model`, display text) this carries the per-model generation
    knobs the final cross-model panel needs. The defaults describe a plain non-reasoning
    chat model that returns choice-token logprobs (the gpt-4o-mini / Gemma case); reasoning
    models override `reasoning=True` (which turns logprobs **off**, because reasoning models
    400 on `logprobs=True`, and widens the token budget) plus an effort or token cap.

    `provider_order` pins OpenRouter routing to specific upstream providers (with fallbacks
    disabled) so the open-weight models are served by a fixed stack and quantization rather
    than whatever backend a request happens to land on - this keeps the Gemma scale ladder a
    clean comparison and caps Qwen's provider.
    """

    key: str
    model: str
    label: str
    note: str = ""
    reasoning: bool = False
    temperature: float = 0.0
    reasoning_effort: str | None = None
    reasoning_tokens: int | None = None
    provider_order: tuple[str, ...] = ()

    def generate_config(self) -> GenerateConfig:
        """The `GenerateConfig` for this model: logprobs + token budget keyed off
        `reasoning`, plus whatever reasoning effort/budget and temperature it needs."""
        cfg: dict = dict(temperature=self.temperature)
        if self.reasoning:
            # Reasoning models reject logprobs and need room for the thinking trace.
            cfg.update(max_tokens=REASONING_MAX_TOKENS, logprobs=False)
        else:
            cfg.update(max_tokens=CHOICE_MAX_TOKENS, logprobs=True, top_logprobs=5)
        if self.reasoning_effort is not None:
            cfg["reasoning_effort"] = self.reasoning_effort
        if self.reasoning_tokens is not None:
            cfg["reasoning_tokens"] = self.reasoning_tokens
        return GenerateConfig(**cfg)

    def model_args(self) -> dict:
        """Extra `get_model` args: an OpenRouter provider pin when `provider_order` is set."""
        if self.provider_order:
            return {
                "provider": {
                    "order": list(self.provider_order),
                    "allow_fallbacks": False,
                }
            }
        return {}


# Ordered. `mock` + `gpt-4o-mini` are the keyless / cheap baselines (gpt-4o-mini results are
# already collected). The rest is the final cross-model panel (see memory `final-model-panel`):
# a reasoning cohort matched on capability across labs, all at their lowest reasoning effort,
# plus a non-reasoning Gemma scale ladder. Provider pins verified against OpenRouter's
# endpoints API: DeepInfra serves the whole Gemma ladder (4B/12B bf16, 27B fp8); Qwen3.7-Plus
# is Alibaba-only. Confirm any model with the `probe` command before a run.
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
        "Cheap; returns choice-token logprobs. Prior baseline (results already collected).",
    ),
    # --- Reasoning panel (lowest non-zero reasoning effort) ---
    ModelOption(
        "claude",
        "openrouter/anthropic/claude-sonnet-4.6",
        "Claude Sonnet 4.6 - OpenRouter",
        "Reasoning panel (Batch 1). Anthropic extended thinking forces temperature=1.",
        reasoning=True,
        temperature=1.0,
        reasoning_tokens=1024,
    ),
    ModelOption(
        "gpt",
        "openrouter/openai/gpt-5.4",
        "GPT-5.4 - OpenRouter",
        "Reasoning panel (Batch 1). OpenAI's current gen is reasoning-only.",
        reasoning=True,
        reasoning_effort="low",
    ),
    ModelOption(
        "gemini",
        "openrouter/google/gemini-3.5-flash",
        "Gemini 3.5 Flash - OpenRouter",
        "Reasoning panel (Batch 1). Multilingual focus; Gemma's closed sibling.",
        reasoning=True,
        reasoning_effort="low",
    ),
    ModelOption(
        "grok",
        "openrouter/x-ai/grok-4.3",
        "Grok 4.3 - OpenRouter",
        "Reasoning panel (Batch 2). Reasoning-only lab.",
        reasoning=True,
        reasoning_effort="low",
    ),
    ModelOption(
        "qwen",
        "openrouter/qwen/qwen3.7-plus",
        "Qwen 3.7 Plus - OpenRouter",
        "Reasoning panel (Batch 2). Chinese/multilingual. Slow; reasoning budget capped.",
        reasoning=True,
        reasoning_tokens=512,
        provider_order=("Alibaba",),
    ),
    # --- Non-reasoning Gemma scale ladder (DeepInfra pinned for a clean comparison) ---
    ModelOption(
        "gemma-4b",
        "openrouter/google/gemma-3-4b-it",
        "Gemma 3 4B - OpenRouter",
        "Scale ladder. DeepInfra bf16 pinned.",
        provider_order=("DeepInfra",),
    ),
    ModelOption(
        "gemma-12b",
        "openrouter/google/gemma-3-12b-it",
        "Gemma 3 12B - OpenRouter",
        "Scale ladder. DeepInfra bf16 pinned.",
        provider_order=("DeepInfra",),
    ),
    ModelOption(
        "gemma-27b",
        "openrouter/google/gemma-3-27b-it",
        "Gemma 3 27B - OpenRouter",
        # No bf16 endpoint at 27B. Both Parasail and DeepInfra serve fp8; Parasail first
        # because DeepInfra's 27B endpoint stalled (no response) during the smoke test while
        # serving 4B/12B fine. Same quant either way, so the ladder comparison is unaffected.
        "Scale ladder. Parasail fp8, DeepInfra fp8 backup.",
        provider_order=("Parasail", "DeepInfra"),
    ),
)

# Named batches for the iterative `sweep` (see the execution plan): Batch 1 first, assess
# time/budget, then Batch 2 tops up the same log dir (`analyze --all` pools across runs).
BATCHES: dict[str, tuple[str, ...]] = {
    "batch1": ("claude", "gpt", "gemini", "gemma-4b", "gemma-12b", "gemma-27b"),
    "batch2": ("grok", "qwen"),
    "gemma": ("gemma-4b", "gemma-12b", "gemma-27b"),
    "reasoning": ("claude", "gpt", "gemini", "grok", "qwen"),
}

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
