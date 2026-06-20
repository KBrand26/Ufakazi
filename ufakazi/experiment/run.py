"""Run the truthiness-bias eval.

Defaults to the keyless deterministic mock so the pipeline runs end-to-end with no API
key. `--model` takes a registry key (`gpt-4o-mini`), the `default` alias, or any raw
Inspect model string. Needs `OPENROUTER_API_KEY` in `.env` for the OpenRouter models.

    uv run python -m ufakazi.experiment.run                      # mock, keyless
    uv run python -m ufakazi.experiment.run --model default      # openrouter/openai/gpt-4o-mini
    uv run python -m ufakazi.experiment.run --model openrouter/anthropic/claude-3.5-haiku
"""

from __future__ import annotations

import argparse

from inspect_ai import eval as inspect_eval

from ufakazi.experiment.models import resolve_model
from ufakazi.experiment.task import truthiness_bias

DEFAULT_LOG_DIR = "results/logs"


def run(
    model: str | None = None,
    log_dir: str = DEFAULT_LOG_DIR,
    languages: tuple[str, ...] = ("en",),
):
    return inspect_eval(
        truthiness_bias(languages=languages),
        model=resolve_model(model),
        log_dir=log_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Ufakazi truthiness-bias eval."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Registry key (e.g. gpt-4o-mini), 'default', or a raw Inspect model "
        "string. Omit for the keyless mock.",
    )
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    args = parser.parse_args()
    run(model=args.model, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
