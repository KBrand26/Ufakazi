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

Headless, config-driven experiment harness. Module boundaries:

- **`scenarios/`** — testimony content as version-controlled data fixtures (English source +
  cached translations), tagged `scenario_id`, `language`, `translation_provenance`. Synthetic
  only.
- **`providers/`** — thin adapter per provider behind one interface
  `generate(messages, system, capture_logprobs) -> Response`. Hides the per-provider logprob
  capability difference (OpenAI yes, Anthropic no). Likely litellm-backed (decision deferred).
- **`experiment/`** — core loop: takes a declarative factor-level config, expands the factorial
  of trials, runs them, parses the forced-choice result.
- **`analysis/`** — preference rates, language main effect, position-bias baseline.
- **`results/`** — gitignored output store (JSONL per trial → flattened parquet/CSV).

A **trial** = `(scenario, language_assignment, position_order, model, system_prompt)`.
Measure = forced binary choice (primary) + choice-token logprob (continuous, where available).
Counterbalancing isolates the language main effect: randomize position, permute language
assignment, and include same-language controls. Results are cached keyed on the trial tuple so
reruns skip completed API calls.

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
- **Ask before adding dependencies** (litellm is the one assumed-but-deferred exception).
- **Data discipline:** all scenario/testimony content is synthetic. No real client or personal
  data anywhere. `results/` and data-file extensions are gitignored.
- **Test for signal, not coverage:** prioritize parsing/normalization, counterbalancing logic,
  and analysis correctness (the bits a wrong result would hide in). Skip framework passthroughs.
- **External-facing writing:** no em dashes, no emojis.
- Build core H1 first; defer H2 and stretch goals until the core loop produces clean results.

## Run / test

```sh
uv run pytest          # tests
uv run ruff check      # lint
uv run ruff format     # format
```

Package is a flat root-level layout (`ufakazi/`, no `src/`), uv build backend. Deferred:
provider API keys via `.env` (gitignored); no GCP deploy target for the hackathon.
