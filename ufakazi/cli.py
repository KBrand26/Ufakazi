"""The `ufakazi` command line: a thin Typer layer over the experiment functions.

Subcommands:
  run         - run the truthiness-bias eval (keyless mock by default)
  probe       - check whether a model parses the choice and returns logprobs
  analyze     - summarize logged trials (baselines, language effect, verbosity confound)
  rationales  - print the model's rationales for inspecting why it chose a language

Model selection has three modes, by design: no `--model` runs the keyless mock; `--model`
takes a registry key / `default` / a raw Inspect string for scripted, reproducible runs;
`--interactive` opens a picker. The explicit `--model` path is what study runs use.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from ufakazi.analysis.load import (
    filter_latest_eval,
    language_preference,
    language_report,
    load_trials,
    per_scenario_language_effect,
    summarize,
)
from ufakazi.analysis.verbosity import (
    effect_vs_length,
    length_bias_controls,
    length_summary,
)
from ufakazi.experiment.probe import probe as probe_model
from ufakazi.experiment.run import run as run_eval
from ufakazi.experiment.trials import DEFAULT_DESIGN
from ufakazi.providers import DEFAULT_KEY, MODEL_REGISTRY

app = typer.Typer(
    help="Ufakazi: language-conditioned truthiness-bias eval harness.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

DEFAULT_LOG_DIR = "results/logs"


def select_model_interactively() -> str:
    """Show the model registry as a table and return the chosen spec (key or raw string)."""
    table = Table(title="Select a model")
    table.add_column("#", justify="right", style="bold")
    table.add_column("key")
    table.add_column("model")
    table.add_column("note", style="dim")
    by_index = {}
    for index, option in enumerate(MODEL_REGISTRY, start=1):
        by_index[str(index)] = option.key
        default_marker = (
            " [green](default)[/green]" if option.key == DEFAULT_KEY else ""
        )
        table.add_row(
            str(index), option.key + default_marker, option.label, option.note
        )
    console.print(table)

    answer = Prompt.ask(
        "Model (number, key, or raw model string)", default=DEFAULT_KEY
    ).strip()
    # A bare index selects that row; otherwise the answer is a key or raw model string.
    return by_index.get(answer, answer)


@app.command()
def run(
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Registry key, 'default', or a raw Inspect model string. "
        "Omit for the keyless mock.",
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Pick a model from a menu."
    ),
    languages: str = typer.Option(
        "en,afr",
        "--languages",
        "-l",
        help="Comma-separated language set, expanded into controls + cross-language trials.",
    ),
    epochs: int = typer.Option(
        10,
        "--epochs",
        "-e",
        help="Replications per trial (samples provider nondeterminism).",
    ),
    design: str = typer.Option(
        DEFAULT_DESIGN,
        "--design",
        "-d",
        help="'star' (each language vs the English reference + human/machine pairs) "
        "or 'full' (complete crossing).",
    ),
    reference: str = typer.Option(
        None,
        "--reference",
        help="Reference language for the star design (default: en, else first language).",
    ),
    log_dir: str = typer.Option(DEFAULT_LOG_DIR, "--log-dir"),
) -> None:
    """Run the truthiness-bias eval (keyless mock by default)."""
    if interactive:
        if model is not None:
            raise typer.BadParameter("Use either --model or --interactive, not both.")
        model = select_model_interactively()
    if design not in ("star", "full"):
        raise typer.BadParameter("--design must be 'star' or 'full'.")
    langs = tuple(lang.strip() for lang in languages.split(",") if lang.strip())
    run_eval(
        model=model,
        log_dir=log_dir,
        languages=langs,
        epochs=epochs,
        design=design,
        reference=reference,
    )


@app.command()
def probe(
    model: str = typer.Argument(
        ..., help="Registry key, 'default', or a raw Inspect model string."
    ),
) -> None:
    """Probe a model: does it parse the choice and return choice-token logprobs?"""
    report = probe_model(model)
    parsed = report["parsed_choice"]
    console.print(f"[bold]model:[/bold]            {report['model']}")
    console.print(
        "[bold]parses CHOICE:[/bold]    "
        + (f"[green]yes ({parsed})[/green]" if parsed else "[red]NO[/red]")
    )
    console.print(
        "[bold]returns logprobs:[/bold] "
        + ("[green]yes[/green]" if report["has_logprobs"] else "[yellow]no[/yellow]")
    )
    if report["serving_provider"]:
        console.print(f"[bold]served by:[/bold]        {report['serving_provider']}")
    console.print(f"[bold]completion:[/bold]       {report['completion']!r}")


@app.command()
def analyze(
    log_dir: str = typer.Option(DEFAULT_LOG_DIR, "--log-dir"),
    all_runs: bool = typer.Option(
        False,
        "--all",
        help="Aggregate every run in the dir (default: latest run only).",
    ),
) -> None:
    """Summarize logged trials: control baselines, then the language main effect."""
    df = load_trials(log_dir)
    if not all_runs:
        df = filter_latest_eval(df)

    summary = summarize(df)
    console.print(
        f"Loaded {summary['n_trials']} trials "
        f"({summary['n_parse_errors']} parse errors)"
    )
    console.print("\n[bold]Same-language controls[/bold] (no language difference):")
    console.print(
        f"  position bias (chose first-presented): {summary['position_first_rate']:.2f}  "
        "[dim](~0.5 = none)[/dim]"
    )
    console.print(
        f"  content preference for testimony A:    {summary['content_pref_A_rate']:.2f}  "
        "[dim](~0.5 = balanced)[/dim]"
    )

    reports = language_report(df)
    if not reports:
        console.print(
            "\n[yellow]No cross-language trials found (single-language run?).[/yellow]"
        )
        return

    console.print("\n[bold]Language main effect[/bold] (cross-language trials):")
    for r in reports:
        lo, hi = r["ci95"]
        verdict = (
            "[green]significant[/green]"
            if r["significant"]
            else "[dim]not significant[/dim]"
        )
        console.print(
            f"  P(prefer {r['target']} over {r['reference']}): "
            f"{r['p_prefer_target']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  {verdict}  "
            f"[dim](n={r['n_cross_trials']})[/dim]"
        )
        if r["p_prefer_target_continuous"] is not None:
            console.print(
                f"      continuous (logprob mass on {r['target']}): "
                f"{r['p_prefer_target_continuous']:.3f}"
            )

    langs_present = set(df["lang_A"].dropna()) | set(df["lang_B"].dropna())
    if {"afr", "afr_mt"} <= langs_present:
        hm = language_preference(df, target="afr_mt", reference="afr")
        lo, hi = hm["ci95"]
        verdict = (
            "[green]significant[/green]"
            if hm["significant"]
            else "[dim]not significant[/dim]"
        )
        console.print(
            "\n[bold]Human vs machine Afrikaans[/bold] "
            "(same content & language, only translation source differs):"
        )
        console.print(
            f"  P(prefer machine over human afr): {hm['p_prefer_target']:.3f}  "
            f"95% CI [{lo:.3f}, {hi:.3f}]  {verdict}  [dim](n={hm['n_cross_trials']})[/dim]"
        )

    target, reference = reports[0]["target"], reports[0]["reference"]
    effects = per_scenario_language_effect(df, target)
    if not effects.empty:
        console.print(
            f"\n[bold]Per-scenario shift[/bold] "
            f"(P(chose A | A={target}) - P(chose A | A={reference})):"
        )
        for _, row in effects.iterrows():
            console.print(f"  {row['scenario_id']:34} {row['language_effect']:+.3f}")

    _print_verbosity(df, target, reference)


def _print_verbosity(df, target: str, reference: str) -> None:
    """Verbosity confound check: is the language effect just a length preference?"""
    lens = length_summary(target, reference)
    if not lens:
        return
    bias = length_bias_controls(df)
    corr = effect_vs_length(df, target, reference)
    console.print(
        "\n[bold]Verbosity confound[/bold] (could length, not language, drive it?):"
    )
    console.print(
        f"  {target} testimonies run [bold]{lens['mean_ratio_target_over_reference']:.2f}x[/bold] "
        f"the length of {reference} "
        f"[dim]({lens[f'mean_len_{target}']:.0f} vs {lens[f'mean_len_{reference}']:.0f} chars, "
        f"{lens['target_longer_fraction']:.0%} longer)[/dim]"
    )
    lo, hi = bias["ci95"]
    verdict = (
        "[green]significant[/green]"
        if bias["significant"]
        else "[dim]not significant[/dim]"
    )
    console.print(
        f"  length bias (controls): P(chose longer) = [bold]{bias['p_chose_longer']:.3f}[/bold]  "
        f"95% CI [{lo:.3f}, {hi:.3f}]  {verdict}  [dim](n={bias['n']})[/dim]"
    )
    console.print(
        f"  per-scenario corr(length advantage, {target} preference): "
        f"r = {corr['pearson_r']:+.2f}  [dim](n={corr['n_scenarios']} scenarios)[/dim]"
    )


@app.command()
def rationales(
    language: str = typer.Option(
        "afr",
        "--language",
        "-l",
        help="Show rationales where the model chose this language.",
    ),
    scenario: str = typer.Option(
        None, "--scenario", "-s", help="Filter to a single scenario id."
    ),
    limit: int = typer.Option(25, "--limit", "-n", help="Max rationales to print."),
    log_dir: str = typer.Option(DEFAULT_LOG_DIR, "--log-dir"),
    all_runs: bool = typer.Option(
        False, "--all", help="Use every run, not just the latest."
    ),
) -> None:
    """Print the model's own rationales for trials where it chose a given language.

    For reading *why* the model favoured the (usually dispreferred) Afrikaans testimony,
    especially in scenarios that buck the overall pattern."""
    df = load_trials(log_dir)
    if not all_runs:
        df = filter_latest_eval(df)
    sub = df[df["valid"] & (df["lang_chosen"] == language)]
    if scenario:
        sub = sub[sub["metadata_scenario_id"] == scenario]
    if "rationale" in sub.columns:
        sub = sub[sub["rationale"].notna()]
    console.print(
        f"[bold]{len(sub)} trials[/bold] where the model chose the [bold]{language}[/bold] "
        f"testimony{f' in {scenario}' if scenario else ''}. Showing up to {limit}:\n"
    )
    for _, row in sub.head(limit).iterrows():
        console.print(
            f"[dim]{row['metadata_scenario_id']} · "
            f"{row['metadata_lang_first']}/{row['metadata_lang_second']} · "
            f"order {row['metadata_position_order']}[/dim]"
        )
        console.print(f"  {row['rationale']}\n")


if __name__ == "__main__":
    app()
