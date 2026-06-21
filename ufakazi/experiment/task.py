"""The Inspect task: dataset of counterbalanced trials, a plain generate solver, and the
record-only scorer. Model and generation config are supplied by the runner / eval call."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from ufakazi.experiment.scoring import record_choice
from ufakazi.experiment.trials import DEFAULT_DESIGN, build_dataset
from ufakazi.providers import generation_config
from ufakazi.scenarios.loader import load_scenarios


@task
def truthiness_bias(
    languages: tuple[str, ...] = ("en", "afr"),
    epochs: int = 10,
    design: str = DEFAULT_DESIGN,
    reference: str | None = None,
    config: GenerateConfig | None = None,
) -> Task:
    """Counterbalanced truthiness-bias task.

    `design` selects which language pairs are rendered: `star` (default, every language
    against the English reference plus human-vs-machine provenance pairs) or `full` (the
    complete crossing). `epochs` replays every trial N times. The provider is materially
    nondeterministic even at temperature 0 (greedy decoding still rides on nondeterministic
    floating-point reductions and routing), so a single call per trial would measure noise;
    the epochs give a choice distribution per condition. No model-call cache is enabled, on
    purpose: caching would collapse the epochs back to one response.

    `config` is the per-model generation config (reasoning/logprob/temperature settings).
    It must match the config the model was built with, because Inspect lets this task
    ("active") config override the model's own; defaults to the logprob-requesting config.
    """
    scenarios = load_scenarios()
    dataset = build_dataset(
        scenarios, languages=tuple(languages), design=design, reference=reference
    )
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=record_choice(),
        config=config or generation_config(),
        epochs=epochs,
    )
