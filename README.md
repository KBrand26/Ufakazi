<p align="center">
  <img src="assets/UfakaziLogo.png" alt="Ufakazi logo" width="180">
</p>

<h1 align="center">Ufakazi</h1>

<p align="center">
  <img src="https://img.shields.io/badge/AI%20Safety-research-5BA893" alt="AI Safety research">
  <img src="https://img.shields.io/badge/status-hackathon%20WIP-E0A24A" alt="Status: hackathon WIP">
  <img src="https://img.shields.io/badge/python-3.14-3776AB?logo=python&logoColor=white" alt="Python 3.14">
  <img src="https://img.shields.io/badge/managed%20with-uv-DE5FE9?logo=uv&logoColor=white" alt="Managed with uv">
  <img src="https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black" alt="Linted with ruff">
  <img src="https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white" alt="Tested with pytest">
</p>

Ufakazi (isiZulu: "witness"). Auditing LLMs for language-based truthiness and legal-context
bias in the South African setting: given two contradicting, evidentially-balanced testimonies,
does a model favor one based on the language (or register) it is written in?

See [`DESIGN.md`](DESIGN.md) for the experimental design and [`CLAUDE.md`](CLAUDE.md) for
architecture and conventions.

Built on [Inspect](https://inspect.aisi.org.uk/) (`inspect_ai`, AISI's eval framework): each
trial is an Inspect `Sample`, and a record-only scorer captures the model's forced choice rather
than grading it (there is no correct answer).

Real models go through [OpenRouter](https://openrouter.ai/) by default (one API key for many
providers); native `openai/` / `anthropic/` strings also work.

## Layout

- `ufakazi/cli.py` — the `ufakazi` command (Typer): `run`, `probe`, `analyze`.
- `ufakazi/scenarios/` — synthetic testimony YAML fixtures (English source + translations) and loader.
- `ufakazi/experiment/` — Inspect task: counterbalanced trial expansion (language set x position, replicated over `epochs`), forced-choice prompt, record-only scorer, runner, model resolver, keyless mock, probe.
- `ufakazi/providers/` — model registry, defaults, and `GenerateConfig` (Inspect handles the provider layer).
- `ufakazi/analysis/` — `samples_df`-based control baselines, the language main effect (cross-language aggregation), per-scenario shift, and a continuous logprob measure, with scenario-level bootstrap CIs.
- `results/` — gitignored experiment output (Inspect `.eval` logs).

## Develop

```sh
uv run ufakazi run                  # end-to-end (keyless mock, en+afr, epochs 10)
uv run ufakazi run --model default  # real default: openrouter/openai/gpt-4o-mini
uv run ufakazi run --model default -l en,afr -e 10   # explicit language set + epochs
uv run ufakazi run --interactive    # pick a model from a menu
uv run ufakazi probe default        # does a model parse + return logprobs?
uv run ufakazi analyze              # control baselines + language main effect (latest run)
uv run inspect view --log-dir results/logs   # browse run outputs in the log viewer
uv run pytest          # tests
uv run ruff check      # lint
uv run ruff format     # format
```

`ufakazi run` with no `--model` needs no API key: it uses Inspect's `mockllm` with a
deterministic responder, so the pipeline runs end-to-end offline. For real models, copy
`.env.example` to `.env` and set `OPENROUTER_API_KEY`.

Each trial is replayed `--epochs` times: the provider is materially nondeterministic even at
temperature 0, so the analysis samples the choice distribution and reports scenario-level
bootstrap confidence intervals rather than a single number.
