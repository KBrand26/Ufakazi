"""One-off model capability probe (output only, no persistence).

Sends a single minimal forced-choice request to a model and reports whether the response
parses and whether choice-token logprobs come back. Use it to vet a model before a study
run, in particular to confirm logprob passthrough via OpenRouter, which varies by the
backend a request is routed to under unpinned routing.

    uv run python -m ufakazi.experiment.probe default
    uv run python -m ufakazi.experiment.probe openrouter/anthropic/claude-3.5-haiku
"""

from __future__ import annotations

import argparse
import asyncio

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, Model

from ufakazi.experiment.models import build_model
from ufakazi.experiment.prompt import SYSTEM_PROMPT
from ufakazi.experiment.scoring import parse_choice

# A minimal, self-contained forced-choice prompt; content is irrelevant to the probe,
# only that the model answers in our CHOICE format and (maybe) returns logprobs.
_PROBE_USER = (
    "Testimony 1:\nThe traffic light was green.\n\n"
    "Testimony 2:\nThe traffic light was red.\n\n"
    "Which testimony is more credible?"
)


def _serving_provider(output) -> str | None:
    """Best-effort: the OpenRouter backend that served the request, if Inspect surfaced it."""
    metadata = output.metadata or {}
    for key in ("provider", "openrouter_provider", "provider_name"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


async def _generate(model: Model):
    return await model.generate(
        [ChatMessageSystem(content=SYSTEM_PROMPT), ChatMessageUser(content=_PROBE_USER)]
    )


def probe(spec: str) -> dict:
    model = build_model(spec)
    output = asyncio.run(_generate(model))
    choices = output.choices
    logprobs = choices[0].logprobs if choices else None
    return {
        "model": model.name,
        "parsed_choice": parse_choice(output.completion),
        "has_logprobs": bool(logprobs and logprobs.content),
        "serving_provider": _serving_provider(output),
        "completion": (output.completion or "").strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a model's capabilities.")
    parser.add_argument(
        "model",
        help="Registry key (e.g. gpt-4o-mini), 'default', or a raw Inspect model string.",
    )
    result = parser.parse_args().model
    report = probe(result)

    parsed = report["parsed_choice"]
    print(f"model:           {report['model']}")
    print(f"parses CHOICE:   {'yes (' + str(parsed) + ')' if parsed else 'NO'}")
    print(f"returns logprobs: {'yes' if report['has_logprobs'] else 'no'}")
    if report["serving_provider"]:
        print(f"served by:       {report['serving_provider']}")
    print(f"completion:      {report['completion']!r}")


if __name__ == "__main__":
    main()
