"""The `ufakazi` command line: a thin Typer layer over the experiment functions.

Subcommands:
  run      - run the truthiness-bias eval (keyless mock by default)
  probe    - check whether a model parses the choice and returns logprobs
  analyze  - summarize logged trials (parse health, position baseline, content pref)

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
    language_report,
    load_trials,
    per_scenario_language_effect,
    summarize,
)
from ufakazi.experiment.probe import probe as probe_model
from ufakazi.experiment.run import run as run_eval
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
    log_dir: str = typer.Option(DEFAULT_LOG_DIR, "--log-dir"),
) -> None:
    """Run the truthiness-bias eval (keyless mock by default)."""
    if interactive:
        if model is not None:
            raise typer.BadParameter("Use either --model or --interactive, not both.")
        model = select_model_interactively()
    langs = tuple(lang.strip() for lang in languages.split(",") if lang.strip())
    run_eval(model=model, log_dir=log_dir, languages=langs, epochs=epochs)


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

    effects = per_scenario_language_effect(df, reports[0]["target"])
    if not effects.empty:
        console.print(
            f"\n[bold]Per-scenario shift[/bold] "
            f"(P(chose A | A={reports[0]['target']}) - P(chose A | A={reports[0]['reference']})):"
        )
        for _, row in effects.iterrows():
            console.print(f"  {row['scenario_id']:34} {row['language_effect']:+.3f}")


if __name__ == "__main__":
    app()
