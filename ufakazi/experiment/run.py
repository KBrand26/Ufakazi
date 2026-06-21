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

from ufakazi.experiment.models import resolve_model
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
):
    return inspect_eval(
        truthiness_bias(
            languages=tuple(languages),
            epochs=epochs,
            design=design,
            reference=reference,
        ),
        model=resolve_model(model),
        log_dir=log_dir,
    )
