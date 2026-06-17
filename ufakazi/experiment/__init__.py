"""The core experiment loop.

Takes a declarative config of factor levels (scenarios, languages, position orders,
models, system prompts), expands the factorial of trials, runs each through a provider,
parses the forced-choice result, and appends one JSONL row per trial. Trials are cached
keyed on their tuple so reruns skip completed API calls.

A trial is ``(scenario, language_assignment, position_order, model, system_prompt)``.
"""
