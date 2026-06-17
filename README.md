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

## Layout

- `ufakazi/scenarios/` — synthetic testimony fixtures (English source + translations).
- `ufakazi/providers/` — model-provider adapters behind one `generate()` interface.
- `ufakazi/experiment/` — config-driven factorial trial loop.
- `ufakazi/analysis/` — preference rates, language main effect, position baseline.
- `results/` — gitignored experiment output.

## Develop

```sh
uv run pytest          # tests
uv run ruff check      # lint
uv run ruff format     # format
```
