# Ufakazi

## Purpose

An AI Safety hackathon project investigating **language-conditioned "truthiness" bias** in
LLMs: given two contradicting, evidentially-balanced testimonies, does a model systematically
favor one based on the *language* (or register) it is written in?

- **H1 (primary):** holding content and position constant, the testimony's language shifts the
  model's credibility preference. Languages: English, Afrikaans, isiXhosa, isiZulu.
- **H2 (secondary, deferred):** prompt language shifts the model's assumed legal framework
  (US/Western vs South African), on cases where SA law diverges.
- **Stretch:** register/style within English — broken English, education/prestige, SA slang.

"Ufakazi" = witness/testimony (isiZulu/isiXhosa). See `DESIGN.md` for the full rationale,
rejected alternatives, and milestone plan. `/new-project` consumes `DESIGN.md` if run later.

## Architecture

Headless harness built on **Inspect** (`inspect_ai`, AISI's eval framework). Inspect provides
the model layer, run loop, logging, and analysis frames; we supply the scenarios, the
counterbalanced trial expansion, the record-only scorer, and the bias analysis. A **trial** is
one Inspect `Sample`. Module boundaries:

- **`scenarios/`** — testimony content as version-controlled YAML fixtures, one `translations`
  entry per language tagged with `provenance` (`source` | `human` | `machine`). `loader.py`
  parses them into `Scenario` / `Testimony` / `Translation`. Synthetic only.
- **`experiment/`** — the Inspect task. `trials.py` expands scenarios into counterbalanced
  `Sample`s (factor levels go into `Sample.metadata`); `prompt.py` is the forced-choice
  template; `scoring.py` is a **record-only `@scorer`** that parses the choice and maps position
  back to content (no target, no correctness); `task.py` wires dataset + `generate()` + scorer;
  `run.py` calls `eval()`; `mock.py` is a deterministic keyless responder for flow tests.
- **`providers/`** — now just model-selection defaults + `GenerateConfig`. Inspect's own model
  layer supersedes the planned per-provider adapter, including logprob capability handling
  (OpenAI yes, Anthropic no); `generation_config()` requests logprobs, ignored where unsupported.
- **`analysis/`** — `load.py` reads logs via Inspect's `samples_df()` into pandas; preference
  rates, position-bias baseline, language main effect.
- **`results/`** — gitignored. Inspect writes one `.eval` log per run under `results/logs/`.

A **trial** = `(scenario, language_assignment, position_order, model, system_prompt)`: model and
system prompt are fixed per `eval()` run; scenario, language assignment, and position order are
encoded per `Sample` (the last two in `metadata`). Measure = forced binary choice (primary) +
choice-token logprob (continuous, where the provider supplies it). Counterbalancing isolates the
language main effect: randomize position, permute language assignment, include same-language
controls. Inspect caches model calls (`cache=True`) and resumes from its logs, covering the
trial-level rerun-skipping the design called for.

## Domain context

- **Validity hinges on translation fidelity.** Testimonies must be content-equivalent across
  languages; this is the central confound, sharpest for isiZulu/isiXhosa (brittle MT, no
  in-team verifier). `translation_provenance` (`human` | `machine`) is a first-class attribute.
- **Afrikaans is the calibration language:** the repo owner is a native speaker, so compare
  human vs machine Afrikaans translations. If machine Afrikaans performs clearly worse, it
  bounds confidence in the isiZulu/isiXhosa results and justifies sourcing an expert speaker.
- **Same-language controls are not optional** — without them, language bias and content
  imbalance are not separable.
- The formal hackathon brief arrives at the event; H2 scope and task framing may shift. The
  config-driven loop is the hedge — every attribute is swappable.

## Working conventions

- **Python via uv only:** `uv run` to execute, `uv add` for deps. Never bare `python`/`pip`.
  Test with pytest; lint/format with ruff. Type-hint function signatures (pragmatic).
- **Ask before adding dependencies.** Current stack: `inspect-ai` (eval framework); `openai`
  (the client Inspect's OpenRouter/OpenAI providers use); `python-dotenv` (load `.env`);
  `pandas` + `pyarrow` (Inspect's `samples_df`); `pyyaml` (fixtures); `typer` + `rich` (the
  `ufakazi` CLI). litellm is not used; Inspect's model layer replaced it.
- **Inspect idioms:** trials are `Sample`s; factor levels live in `Sample.metadata`; the scorer
  records rather than grades (no `target`); read results back with `samples_df()`, not by parsing
  raw logs. Keep the metadata key contract (`trials.py`) in sync with the scorer and `analysis/`.
- **Data discipline:** all scenario/testimony content is synthetic. No real client or personal
  data anywhere. `results/` and data-file extensions are gitignored.
- **Test for signal, not coverage:** prioritize parsing/normalization, counterbalancing logic,
  and analysis correctness (the bits a wrong result would hide in). Skip framework passthroughs.
- **External-facing writing:** no em dashes, no emojis.
- Build core H1 first; defer H2 and stretch goals until the core loop produces clean results.

## Run / test

```sh
uv run ufakazi run                  # run the eval (keyless mock by default)
uv run ufakazi run --model default  # real default: openrouter/openai/gpt-4o-mini
uv run ufakazi run --interactive    # pick a model from a Rich menu
uv run ufakazi probe default        # check a model parses + returns logprobs (no persistence)
uv run ufakazi analyze              # summarize logged trials
uv run pytest          # tests
uv run ruff check      # lint
uv run ruff format     # format
```

The `ufakazi` CLI (Typer, see `cli.py`) is the single entry point. Model selection has three
modes: no `--model` runs the keyless `mockllm` mock (deterministic position-biased responder in
`experiment/mock.py`) so the pipeline runs end-to-end with no API key or spend; `--model` takes a
registry key (`providers.MODEL_REGISTRY`), `default`, or a raw Inspect string (the scriptable path
for study runs); `--interactive` opens a picker. Real models go through **OpenRouter** by default
(one key); put `OPENROUTER_API_KEY` in `.env` (gitignored, auto-loaded in `ufakazi/__init__.py`;
see `.env.example`). Routing is currently unpinned, so logprob availability can vary by backend;
`probe` a model before trusting its logprobs. A benign `Unable to convert value to float: a/b`
warning appears each run (Inspect's epoch reducer averaging our categorical A/B score, which we
never use; analysis reads `samples_df`). Package is flat-root layout (`ufakazi/`, no `src/`).
No GCP deploy target for the hackathon.
