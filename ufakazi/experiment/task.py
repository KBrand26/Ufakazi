"""The Inspect task: dataset of counterbalanced trials, a plain generate solver, and the
record-only scorer. Model and generation config are supplied by the runner / eval call."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.solver import generate

from ufakazi.experiment.scoring import record_choice
from ufakazi.experiment.trials import build_dataset
from ufakazi.providers import generation_config
from ufakazi.scenarios.loader import load_scenarios


@task
def truthiness_bias(
    languages: tuple[str, ...] = ("en", "afr"), epochs: int = 10
) -> Task:
    """Counterbalanced truthiness-bias task.

    `epochs` replays every trial N times. The provider is materially nondeterministic
    even at temperature 0 (greedy decoding still rides on nondeterministic floating-point
    reductions and routing), so a single call per trial would measure noise; the epochs
    give a choice distribution per condition. No model-call cache is enabled, on purpose:
    caching would collapse the epochs back to one response.
    """
    scenarios = load_scenarios()
    dataset = build_dataset(scenarios, languages=tuple(languages))
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=record_choice(),
        config=generation_config(),
        epochs=epochs,
    )
