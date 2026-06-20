"""Run the truthiness-bias eval.

Defaults to the keyless deterministic mock so the pipeline runs end-to-end with no API
key. Pass `--model` (e.g. `openai/gpt-4o-mini`) once provider keys are configured.

    uv run python -m ufakazi.experiment.run                 # mock, keyless
    uv run python -m ufakazi.experiment.run --model openai/gpt-4o-mini
"""

from __future__ import annotations

import argparse

from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model

from ufakazi.experiment.mock import position_biased_responder
from ufakazi.experiment.task import truthiness_bias

DEFAULT_LOG_DIR = "results/logs"


def run(
    model: str | None = None,
    log_dir: str = DEFAULT_LOG_DIR,
    languages: tuple[str, ...] = ("en",),
):
    resolved = model or get_model(
        "mockllm/model", custom_outputs=position_biased_responder(1)
    )
    return inspect_eval(
        truthiness_bias(languages=languages), model=resolved, log_dir=log_dir
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Ufakazi truthiness-bias eval."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Inspect model string (e.g. openai/gpt-4o-mini). Omit for the keyless mock.",
    )
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    args = parser.parse_args()
    run(model=args.model, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
