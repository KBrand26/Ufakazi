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
- **Counterbalancing to isolate the language main effect:**
  - randomize **position** (which testimony appears first) — controls known LLM position bias;
  - **permute language assignment** — same scenario run with A/B in swapped languages;
  - **same-language controls** (both testimonies in one language, orders swapped) — the
    baseline that should show only content/position effects, no language effect. Without it
    language bias and content imbalance are not separable.

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

Headless, modular, plug-and-play. Module boundaries:

- **`scenarios/`** — testimony content as version-controlled data fixtures (English source +
  translations), tagged `scenario_id`, `language`, `translation_provenance`. Synthetic only.
- **`providers/`** — thin adapter per provider behind one interface
  `generate(messages, system, capture_logprobs) -> Response`. Hides the
  Anthropic-no-logprobs / OpenAI-yes-logprobs difference behind a capability flag. Likely
  backed by **litellm** (decision deferred to the hackathon; fallback is hand-rolled adapters).
- **`experiment/`** — the core loop: takes a declarative config of factor levels, expands the
  **factorial of trials**, runs them, parses the choice.
- **`analysis/`** — loads results; preference rates, language main effects, position baseline.
  Notebook or thin module.
- **`results/`** — gitignored output store.

### Persistence

- One **JSONL row per trial** (full tuple + raw response + parsed choice + logprob).
  Append-only → crash-safe and resumable.
- Flatten step → parquet/CSV for analysis.
- **Cache keyed on the trial tuple** so reruns skip completed API calls (avoids re-paying for
  hundreds of calls while iterating). SQLite judged overkill.

### Config-driven runs

Experiments declared in YAML/dict listing factor levels (languages, scenarios, models, system
prompts, position orders). The loop expands and runs — tweak attributes without touching code.

## Key decisions & rejected alternatives

| Decision | Choice | Rejected | Why |
|---|---|---|---|
| Response format | Forced binary choice + choice-token logprob | Likert 1–7 | Clean stats *and* continuous sensitivity; avoids meaningless 3-vs-4 noise. Likert is a later add-on. |
| Scenario source | Hand-authored English first | LLM-generated | Control over evidential balance; LLM-authored added later as a comparison arm. |
| Translation | Cached fixtures, provenance-tagged, Afrikaans as control | Live per-run MT | Reproducible; quantifies the translation confound instead of hiding it. |
| Provider layer | litellm (deferred) behind a thin interface | Hand-rolled SDK adapters | Easy provider swapping; one dep vs. per-SDK logprob handling. Final call at hackathon. |
| Persistence | JSONL + flatten to parquet/CSV, tuple-keyed cache | SQLite / CSV-only | Crash-safe, resumable, cheap reruns; SQLite overkill for the scale. |
| Concurrency | Sequential + cache, async only if needed | asyncio upfront | Don't solve a speed problem that doesn't exist yet. |

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

1. **Skeleton:** 2–3 English scenarios, one provider, hardcoded single trial end-to-end →
   parsed choice persisted to JSONL.
2. **Counterbalancing:** position swap + same-language controls; analysis computes preference
   rate + position baseline.
3. **Languages:** add Afrikaans (human + machine) and the translation fixture pipeline; run the
   calibration comparison.
4. **Scale H1:** isiXhosa + isiZulu, full scenario set, factorial config, tuple-keyed caching.
5. **Iterate:** second provider, LLM-authored scenarios, system-prompt-language variation.
6. **(If in scope) H2:** legal-divergence scenarios + framework-shift analysis.
7. **Report:** charts + writeup for judging.
