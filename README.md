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
does a model favor one based on the language it is written in?

The languages under test are English, Afrikaans, isiXhosa, and isiZulu. Holding content and
position constant, we render the same two testimonies into different languages and measure the
model's forced credibility choice. Counterbalancing cancels content and position bias structurally,
so what remains is the **language main effect**.

See [`DESIGN.md`](DESIGN.md) for the experimental design and [`CLAUDE.md`](CLAUDE.md) for
architecture and conventions.

Built on [Inspect](https://inspect.aisi.org.uk/) (`inspect_ai`, AISI's eval framework): each
trial is an Inspect `Sample`, and a record-only scorer captures the model's forced choice rather
than grading it (there is no correct answer).

Real models go through [OpenRouter](https://openrouter.ai/) by default (one API key for many
providers); native `openai/` / `anthropic/` strings also work.

## Headline findings

Interim panel of 8 models, 20 synthetic scenarios, full star design, 5 epochs (top-up to 10 was
deemed too costly for the hackathon, so confidence intervals are wide-ish and these are not final):

- **Only Claude Sonnet 4.6 and GPT-5.4 are bias-robust.** Their credibility preference is flat at
  chance across all four languages.
- **Every other model tested shows significant pro-English bias**: Gemini 3.5 Flash, the full Gemma
  3 ladder (4B / 12B / 27B), Grok 4.3, and (mildly) Qwen 3.7 Plus. The bias is **not** explained by
  capability (Gemini and Grok are frontier), by scale (it does not shrink across the Gemma ladder),
  or by a single lab (it spans Google, xAI, and Alibaba).
- **A cross-linguistic gradient holds for every biased model**: Afrikaans bias is mild, isiXhosa and
  isiZulu severe. Human and machine Afrikaans agree closely, so machine translation is not the driver.
- **Bias can override the evidence.** On scenarios where a model otherwise picks one testimony almost
  every time, rewriting that same testimony in isiZulu flips the choice for Gemma 27B 92% of the time
  and Grok 78% of the time. Claude and GPT essentially never flip.
- **The rationales are a smoking gun**: biased models explicitly justify favoring English ("native
  language, more accurate recollection", "primary language of the proceedings"). Claude mentions
  language the most, but to translate and engage the other testimony's content; GPT is largely
  language-blind. Two routes to fairness: read-and-neutralize, or ignore.

Figures and tables behind these numbers are produced by `ufakazi figures` (see below).

## Layout

- `ufakazi/cli.py` — the `ufakazi` command (Typer): `run`, `sweep`, `probe`, `analyze`, `rationales`, `figures`.
- `ufakazi/scenarios/` — synthetic testimony YAML fixtures (English source + per-language translations, each tagged `source` / `human` / `machine`) and loader.
- `ufakazi/experiment/` — Inspect task: counterbalanced trial expansion (language set x position, replicated over `epochs`), forced-choice prompt, record-only scorer, runner, per-model resolver, keyless mock, probe.
- `ufakazi/providers/` — model registry, per-model generation config (reasoning vs logprobs), and OpenRouter provider pins.
- `ufakazi/analysis/` — `samples_df`-based, model-aware analysis with scenario-level bootstrap CIs. Two arms, split per model from each model's own controls: **balanced** scenarios feed the clean language main effect; **saturated** scenarios (a clear content favourite) feed the content-override probe.
- `ufakazi/figures.py` — programmatic paper figures (vector PDF + PNG) and tidy tables (CSV + `macros.tex`) from the logs; matplotlib + seaborn, lazily imported.
- `results/` — gitignored experiment output: Inspect `.eval` logs under `results/logs/`, figures and tables under `results/figures/`.

## Two-arm analysis

Counterbalancing isolates the language main effect, but it only does so cleanly when the two
contents are roughly evidence-balanced for a given model. Some scenarios are not: the model has a
clear favourite regardless of language. Rather than discard those, we use them.

Each model's same-language controls measure a per-scenario `content_skew`. Scenarios above the
saturation threshold are split off into a second arm. **Balanced** scenarios give the clean
**language main effect** (`P(prefer the target-language testimony)`, 0.5 = no bias). **Saturated**
scenarios give the **content-override probe**: when a model almost always picks one testimony,
does rewriting it in another language flip that choice? The split is dynamic, computed per model,
so a scenario that saturates a weak model can stay balanced for a strong one. Keep the scenario set
diverse; do not engineer balance.

## Develop

```sh
uv run ufakazi run                  # end-to-end (keyless mock, en+afr, epochs 10)
uv run ufakazi run --model default  # real default: openrouter/openai/gpt-4o-mini
uv run ufakazi run --model default -l en,afr -e 10   # explicit language set + epochs
uv run ufakazi run --interactive    # pick a model from a menu
uv run ufakazi sweep -m batch1 -e 1 --limit 8        # smoke-test the panel (~$0.50, 1 epoch)
uv run ufakazi sweep -m batch1      # run a whole panel: one eval per model into a shared log dir
uv run ufakazi probe default        # does a model parse + return logprobs?
uv run ufakazi analyze              # per model: controls, balance split, main effect + override probe
uv run ufakazi analyze --all        # pool every run per model (epochs are additive)
uv run ufakazi rationales claude    # the model's own reasons on trials where it chose a language
uv run ufakazi figures              # paper figures (PDF + PNG) + tables -> results/figures
uv run inspect view --log-dir results/logs   # browse run outputs in the log viewer
uv run pytest          # tests
uv run ruff check      # lint
uv run ruff format     # format
```

`ufakazi run` with no `--model` needs no API key: it uses Inspect's `mockllm` with a
deterministic responder, so the pipeline runs end-to-end offline. For real models, copy
`.env.example` to `.env` and set `OPENROUTER_API_KEY`. `sweep` runs several models in sequence
(one `.eval` per model) and tops up the same log dir, so a panel is built iteratively; `analyze`
and `figures` loop per model and read the `model` column.

Each trial is replayed `--epochs` times: the provider is materially nondeterministic even at
temperature 0, so the analysis samples the choice distribution and reports scenario-level
bootstrap confidence intervals rather than a single number. Epochs across separate runs of the
same model pool additively, so a panel can be deepened run by run rather than in one expensive shot.
