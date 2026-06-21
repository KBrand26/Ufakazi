"""Run the truthiness-bias eval.

Defaults to the keyless deterministic mock so the pipeline runs end-to-end with no API
key. `--model` takes a registry key (`gpt-4o-mini`), the `default` alias, or any raw
Inspect model string. Needs `OPENROUTER_API_KEY` in `.env` for the OpenRouter models.

Languages default to the `en`/`afr` calibration set (expanded into same-language controls
plus cross-language trials), replicated `epochs` times to sample the provider's residual
nondeterminism.

    uv run ufakazi run                      # mock, keyless
    uv run ufakazi run --model default      # openrouter/openai/gpt-4o-mini, en+afr
    uv run ufakazi run --model default --languages en,afr --epochs 10
"""

from __future__ import annotations

from inspect_ai import eval as inspect_eval

from ufakazi.experiment.models import resolve_model, task_config_for
from ufakazi.experiment.task import truthiness_bias
from ufakazi.experiment.trials import DEFAULT_DESIGN

DEFAULT_LOG_DIR = "results/logs"


def run(
    model: str | None = None,
    log_dir: str = DEFAULT_LOG_DIR,
    languages: tuple[str, ...] = ("en", "afr"),
    epochs: int = 10,
    design: str = DEFAULT_DESIGN,
    reference: str | None = None,
    max_connections: int | None = None,
    limit: int | None = None,
):
    """Run the eval for a single model.

    The per-model generation config (`task_config_for`) is threaded into the task so the
    model is called with its own reasoning/logprob settings rather than a global default.
    `max_connections` raises in-run concurrency (helps the slow reasoning models);
    `limit` caps the number of samples (the smoke-test lever).
    """
    # `eval()` GenerateConfigArgs (max_connections) and `limit` are only forwarded when set,
    # so the defaults stay Inspect's own.
    extra: dict = {}
    if max_connections is not None:
        extra["max_connections"] = max_connections
    if limit is not None:
        extra["limit"] = limit
    return inspect_eval(
        truthiness_bias(
            languages=tuple(languages),
            epochs=epochs,
            design=design,
            reference=reference,
            config=task_config_for(model),
        ),
        model=resolve_model(model),
        log_dir=log_dir,
        **extra,
    )
