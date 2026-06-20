# Ufakazi — Design

*AI Safety hackathon project. "Ufakazi" = witness/testimony (isiZulu/isiXhosa).*

## Goal

Investigate whether LLMs exhibit a **language-conditioned "truthiness" bias**: given two
contradicting testimonies that are equivalent in evidential weight, does a model
systematically favor one based on the *language* it is written in?

A secondary, deferred question (H2): does switching the prompt/system language shift a
model's operating legal assumptions from US/Western defaults toward South African law, on
cases where SA law diverges. Decided once the full hackathon brief lands.

### Hypotheses

- **H1 (primary) — truthiness bias:** holding content and position constant, the language a
  testimony is written in shifts the model's credibility preference.
- **H2 (secondary, deferred) — constitutional/cultural shift:** prompt language shifts the
  model's assumed legal framework (US/Western vs South African).

### Users / context

Two-person hackathon team. Both native English speakers; one (repo owner) is a native
Afrikaans speaker and can produce reliable human Afrikaans translations.

### Success / demo moment

A **reproducible results notebook + short report with charts**: quantified, counterbalanced
preference rates per language with a position-bias baseline, plus the Afrikaans
human-vs-machine translation control that bounds confidence in the low-resource-language
results. Live dashboard explicitly out of scope.

## Core-first scope

**Must work end-to-end first (the walking skeleton):**

1. A small set of hand-authored English scenarios (~5–10), each two contradicting,
   evidentially balanced testimonies.
2. Translations into the target languages, stored as cached fixtures.
3. A config-driven experiment loop that expands the factorial of trials and runs them
   through one provider.
4. Parsing + persistence of each trial's choice (and logprob where available).
5. Analysis producing preference rate, language main effect, and the position-bias baseline.

**Deferred (earned later, as the brief stabilizes):**

- H2 legal-framework track.
- LLM-authored scenarios (compared against hand-authored).
- Likert credibility ratings (logprobs cover continuous signal first).
- Additional providers, model sizes, system-prompt-language variation.
- Register/style stretch goals (broken English, education/prestige, SA slang) — see below.
- Concurrency/async — sequential + caching until it hurts.

## Experimental design

A **trial** is fully specified by the tuple:

```
(scenario, language_assignment, position_order, model, system_prompt)
```

- **Task:** forced binary choice — "which testimony is more credible/truthful?" plus a
  one-line rationale.
- **Primary measure:** preference rate (clean stats). **Continuous measure:** logprob of the
  choice token → smooth 0–1 sentiment, sensitive to subtle shifts without Likert noise.
  Logprobs are a per-provider capability (OpenAI yes, Anthropic no); binary choice is the
  universal fallback.
- **Languages are a set.** For a language set L, each scenario expands to every assignment of
  languages to its two *contents* (|L|^2) crossed with both position orders. Equal-language
  assignments are same-language controls; differing ones are cross-language trials.
- **Counterbalancing to isolate the language main effect:**
  - randomize **position** (which testimony appears first) — controls known LLM position bias;
  - **permute language assignment over content** — language tracks the testimony identity, not
    the slot, so each content appears in each language equally across the cross trials;
  - **same-language controls** (both testimonies in one language, orders swapped) — the
    baseline that should show only content/position effects, no language effect. Without it
    language bias and content imbalance are not separable.
  - Aggregating the cross trials symmetrically over which content is in which language **cancels
    content bias**, leaving `P(prefer the target-written testimony)` vs 0.5 as the effect.
- **Replication via `epochs` (default 10).** The provider is materially nondeterministic even
  at temperature 0 (see below), so each trial is replayed N times to estimate a choice
  distribution. Significance uses a **scenario-level (cluster) bootstrap**: scenarios are the
  unit of independence, so we resample scenarios with replacement for CIs that respect the
  nesting of epochs within trial within scenario.

### Determinism is not available (measured)

We tested whether `gpt-4o-mini` is reproducible at temperature 0 via OpenRouter. It is not:
15 identical calls split ~60/40 across the two choices, with the choice-token confidence
swinging between p=0.50 and p=0.90. Pinning the OpenRouter provider (`order: [openai]`,
no fallbacks) and adding an OpenAI `seed` did **not** restore determinism — `gpt-4o-mini` is
served by OpenAI either way, and the noise is OpenAI-side (batch-dependent floating-point
reduction order, MoE routing); OpenAI's `seed` is best-effort and cannot override it.
Consequence: we do **not** try to engineer the stochasticity away. We keep temperature at 0
(the model's considered judgment, with variance for free), replicate with epochs, and report
distributional CIs. Model-call caching is deliberately **off** — it would collapse the epochs.

### Translation validity (biggest risk)

H1 rests on testimonies being content-equivalent across languages. Translation quality is the
key confound, sharpest for isiZulu/isiXhosa (brittle machine translation, no in-team verifier).

- Translation is an explicit, **swappable, cached pipeline step** — fixtures, not live
  per-run generation. Reproducible and source-agnostic.
- **`translation_provenance` (`human` | `machine`) is a first-class attribute of every
  testimony.**
- **Afrikaans is the calibration language:** compare the repo owner's *human* Afrikaans
  translations against *machine* Afrikaans. If machine Afrikaans performs clearly worse, that
  bounds (and casts doubt on) the isiZulu/isiXhosa results and justifies sourcing an expert
  speaker.

## Architecture

Headless, modular, plug-and-play, built on **Inspect** (`inspect_ai`, AISI's open eval
framework). Inspect was chosen over a hand-rolled loop because it already provides the model
abstraction, run loop, logging, caching/resume, and analysis frames, and because using an AI
safety institution's standard tooling fits the hackathon. The design maps onto Inspect cleanly:
a **trial is one Inspect `Sample`**, and the forced-choice task needs **no answer label** since
Inspect makes both `target` and scoring optional. Module boundaries:

- **`scenarios/`** — testimony content as version-controlled **YAML fixtures**, one
  `translations` entry per language tagged with `provenance` (`source` | `human` | `machine`).
  `loader.py` parses them into `Scenario` / `Testimony` / `Translation`. Synthetic only.
- **`experiment/`** — the Inspect task. `trials.py` expands scenarios into the **factorial of
  counterbalanced `Sample`s** (language set x position order) with every factor level in
  `Sample.metadata`; `prompt.py` is the forced-choice template; `scoring.py` is a **record-only
  `@scorer`** (parses the choice and maps position back to content; records, does not grade);
  `task.py` wires dataset + `generate()` + scorer + `epochs`; `run.py` calls `eval()`; `mock.py`
  is a deterministic keyless responder.
- **`providers/`** — reduced to model-selection defaults + `GenerateConfig`. Inspect's model
  layer (20+ providers) replaced the planned adapter and handles the logprob capability
  difference (OpenAI yes, Anthropic no) for us.
- **`analysis/`** — loads logs via Inspect's `samples_df()` (one row per trial-epoch) into
  pandas: control baselines, the language main effect via symmetric cross-language aggregation,
  per-scenario shift, and a continuous logprob measure, all with scenario-level bootstrap CIs.
- **`results/`** — gitignored. Inspect writes one `.eval` log per run under `results/logs/`.

### Persistence

- Inspect's native **`.eval` log** is the per-trial store: it records the full sample (input,
  metadata, model output, logprobs, score) and is crash-safe and resumable by design, replacing
  the planned hand-rolled JSONL-then-flatten step.
- Analysis reads logs back with **`samples_df()`** (one row per trial, metadata expanded to
  `metadata_*` columns) rather than parsing raw logs; flatten to parquet/CSV only if needed.
- **Model-call caching is deliberately off.** It would collapse the epoch replications (identical
  input + config) back to a single cached response, defeating the whole point of sampling the
  provider's nondeterminism. Reruns are cheap on `gpt-4o-mini`; log-based resume still covers
  interrupted runs.

### Config-driven runs

Factor levels (languages, scenarios, position orders) feed `trials.py`, which expands them into
the `Sample` list; model and system prompt are arguments to the `eval()` run. The task is
parameterized (e.g. `truthiness_bias(languages=...)`), and Inspect tasks also take CLI `-T`
params, so attributes stay swappable without touching the loop. Inspect does not automate the
factorial or counterbalancing; that logic is ours in `trials.py` (and is the most-tested code).

## Key decisions & rejected alternatives

| Decision | Choice | Rejected | Why |
|---|---|---|---|
| Response format | Forced binary choice + choice-token logprob | Likert 1–7 | Clean stats *and* continuous sensitivity; avoids meaningless 3-vs-4 noise. Likert is a later add-on. |
| Scenario source | Hand-authored English first | LLM-generated | Control over evidential balance; LLM-authored added later as a comparison arm. |
| Translation | Cached fixtures, provenance-tagged, Afrikaans as control | Live per-run MT | Reproducible; quantifies the translation confound instead of hiding it. |
| Framework | **Inspect (`inspect_ai`)** | Hand-rolled loop | Batteries-included model layer, logging, caching/resume, and `samples_df` analysis; AISI tooling fits the safety brief. No-label fit confirmed: `target` and scoring are optional. |
| Provider layer | Inspect model API (20+ providers) | litellm / hand-rolled SDK adapters | Inspect already hides the per-provider logprob difference; one fewer dep and abstraction to own. |
| Scoring | Record-only `@scorer` (no `target`) | Graded scorer / external parse | We measure a choice, not correctness; the scorer records position + content + logprob, metrics are sanity readouts, real stats live in `analysis/`. |
| Persistence | Inspect `.eval` log + `samples_df()` | JSONL + flatten / SQLite | Native, crash-safe, resumable; `samples_df` gives a tidy frame directly. |
| Replication | `epochs` (default 10) + scenario bootstrap | Single call; raise temperature | Temp-0 nondeterminism is real and unavoidable (measured); epochs sample it, bootstrap respects scenario clustering. Caching off so epochs don't collapse. |
| Significance | Scenario-level (cluster) bootstrap | scipy / statsmodels | Respects the epoch-in-trial-in-scenario nesting with no new dependency; formal mixed-effects model deferred. |
| Concurrency | Sequential, async only if needed | asyncio upfront | Don't solve a speed problem that doesn't exist yet. |

## Stretch goals — register/style within a language

A second axis beyond language identity: **sociolinguistic register**. These probe whether the
bias tracks the *language* of a testimony or the *markers of competence, class, and origin*
carried by its style. They fit the harness cleanly as a `register` attribute on a testimony
(same counterbalancing, same forced-choice measure) rather than a new experiment.

- **Broken English** — degraded/non-native English, simulating a non-native speaker testifying
  in English. Tests whether disfluency alone lowers perceived credibility.
- **Education / prestige** — impressive, highly educated phrasing vs. plain common English of
  identical content. Tests a competence-signalling bias.
- **South African slang** — testimony in SA vernacular/slang. Tests whether locally-marked
  informal English is penalized (and interacts interestingly with the H2 SA-context track).

Held to the same validity bar as translation: register variants must preserve content, and the
same-language control still anchors the position/content baseline. Sequenced after the core H1
language results land.

## Risks & unknowns

- **Translation fidelity** (primary threat to H1) — mitigated by the Afrikaans human/machine
  control; expert speaker sourced if the control fails.
- **Scenario balance** — testimonies may not be truly evidentially equal; the same-language
  control surfaces residual content/position imbalance.
- **Hackathon brief** — formal guidelines arrive at the event; H2 scope and possibly the task
  framing may change. The config-driven loop is the hedge: attributes are swappable.
- **Provider capability drift** — logprob availability varies; binary choice keeps every
  provider comparable.

## Game plan (milestones)

1. **M1 — walking skeleton (done):** one English scenario, position counterbalanced as a
   same-language control, run end-to-end through the Inspect task on the keyless mock; record-only
   scorer → `.eval` log → `samples_df` analysis recovering the position baseline. Naive by design;
   proves the pipeline shape.
2. **M2 — OpenRouter provider + CLI (done):** real runs on `openrouter/openai/gpt-4o-mini` with
   choice-token logprobs captured; single `ufakazi` Typer CLI (`run` / `probe` / `analyze`).
3. **M3 — languages: en/afr (done):** ~11 scenarios each with `en` (source) + `afr` (human)
   testimonies; `trials.py` expands the language set into same-language controls + cross-language
   trials; `epochs` replication; analysis reports the language main effect with bootstrap CIs.
   Empirically established that temp-0 determinism is unavailable and pinning does not help, so
   replication is mandatory. First read on `gpt-4o-mini`: a significant preference for English.
   *Next within M3:* add the **machine-Afrikaans** arm (the calibration control vs human afr).
4. **Scale H1:** isiXhosa + isiZulu translations, larger language set in the same expansion.
5. **Iterate:** second provider, LLM-authored scenarios, system-prompt-language variation.
6. **(If in scope) H2:** legal-divergence scenarios + framework-shift analysis.
7. **Report:** charts + writeup for judging (Inspect View for transcript inspection).
