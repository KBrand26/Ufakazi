"""Paper figures: programmatic, reproducible plots for the write-up.

Given a run's `.eval` logs, `generate_figures` deterministically rebuilds every figure plus
the underlying numbers, so the paper's plots and in-text stats regenerate from one command
with no manual steps. Figures are vector **PDF** (for `\\includegraphics` in LaTeX/Overleaf)
with a **PNG** alongside for quick viewing; the tidy tables are also written as CSV and a
LaTeX `macros.tex` so inline numbers stay in sync with the plots.

Matplotlib + seaborn are imported lazily inside `generate_figures` so the rest of the CLI
(run / analyze / probe) works without the figure dependencies installed.

The figure set (priority order), all driven by the per-model tables in `analysis.load`:
  1. forest    - language main effect per model, with cluster-bootstrap 95% CIs (headline)
  2. heatmap   - model x language preference overview
  3. gradient  - effect ordered by language resource level, one line per model
  4. provenance- human vs machine translation, per model (validity anchor)
  5. scale     - the Gemma size ladder, if those models are present
  6. controls  - position / content baselines per model (counterbalancing check)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ufakazi.analysis.load import (
    control_table,
    language_effect_table,
    load_trials,
    models_in,
    provenance_effect_table,
)

# Languages ordered by resource level (English reference -> lowest-resource MT), used for a
# consistent left-to-right / top-to-bottom ordering and labelling across every figure.
LANG_ORDER = ["en", "afr", "afr_mt", "zul_mt", "xho_mt"]
LANG_LABELS = {
    "en": "English",
    "afr": "Afrikaans (human)",
    "afr_mt": "Afrikaans (MT)",
    "zul_mt": "isiZulu (MT)",
    "xho_mt": "isiXhosa (MT)",
}
# Gemma scale ladder: model-string substring -> parameter count (billions).
GEMMA_SIZES = {"gemma-3-4b": 4, "gemma-3-12b": 12, "gemma-3-27b": 27}

NEUTRAL = 0.5  # no-preference line: P(prefer target) = 0.5

# Our own clean, professional style (no journal house-style dependency).
PAPER_STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


def short_model_name(model: str) -> str:
    """Human-friendly label: drop the provider path, keep the model id (`gpt-4o-mini`)."""
    tail = model.split("/")[-1]
    return "mock" if model.startswith("mockllm") else tail


def lang_label(code: str, *, multiline: bool = False) -> str:
    label = LANG_LABELS.get(code, code)
    return label.replace(" (", "\n(") if multiline else label


def _ordered_targets(codes) -> list[str]:
    """Target codes present, ordered by resource level (unknown codes sort to the end)."""
    present = set(codes)
    ranked = [c for c in LANG_ORDER if c in present]
    return ranked + sorted(present - set(ranked))


def _gemma_size(model: str) -> int | None:
    for key, size in GEMMA_SIZES.items():
        if key in model:
            return size
    return None


def _model_palette(sns, models: list[str]) -> dict[str, tuple]:
    colors = sns.color_palette("colorblind", n_colors=max(len(models), 1))
    return {m: colors[i] for i, m in enumerate(models)}


# --- individual figures (each takes already-computed tables + plotting handles) ---


def _forest(plt, palette, effect: pd.DataFrame):
    """Grouped forest plot: P(prefer target) with 95% CI, one marker per model per language."""
    targets = _ordered_targets(effect["target"].unique())
    models = list(palette)
    fig, ax = plt.subplots(
        figsize=(6.4, 0.9 + 0.55 * len(targets) * max(len(models), 1))
    )
    # Dodge models within each language band.
    band = 1.0
    span = band * 0.7
    offsets = (
        [0.0]
        if len(models) == 1
        else [(-span / 2) + span * i / (len(models) - 1) for i in range(len(models))]
    )
    yticks, ylabels = [], []
    for ti, target in enumerate(targets):
        base = ti * band
        yticks.append(base)
        ylabels.append(lang_label(target))
        for mi, model in enumerate(models):
            row = effect[(effect["model"] == model) & (effect["target"] == target)]
            if row.empty:
                continue
            r = row.iloc[0]
            y = base + offsets[mi]
            ax.plot(
                [r["ci_lo"], r["ci_hi"]],
                [y, y],
                color=palette[model],
                lw=1.6,
                alpha=0.9,
                solid_capstyle="round",
            )
            ax.plot(
                r["p_prefer_target"],
                y,
                "o",
                color=palette[model],
                markersize=5.5,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=short_model_name(model) if ti == 0 else None,
            )
    ax.axvline(NEUTRAL, color="0.4", lw=1.0, ls="--", zorder=0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(prefer the target-language testimony over English)")
    ax.set_title("Language main effect by model")
    ax.text(
        NEUTRAL,
        ax.get_ylim()[0],
        "  no preference",
        color="0.4",
        fontsize=8,
        va="bottom",
    )
    if len(models) > 1:
        ax.legend(title="model", loc="best")
    return fig


def _heatmap(plt, sns, effect: pd.DataFrame):
    """Model x language heatmap of P(prefer target), diverging around 0.5."""
    targets = _ordered_targets(effect["target"].unique())
    models = models_in_table(effect)
    grid = effect.pivot_table(
        index="model", columns="target", values="p_prefer_target"
    ).reindex(index=models, columns=targets)
    grid.index = [short_model_name(m) for m in grid.index]
    grid.columns = [lang_label(c, multiline=True) for c in grid.columns]
    fig, ax = plt.subplots(
        figsize=(1.6 + 1.15 * len(targets), 1.4 + 0.5 * max(len(models), 1))
    )
    sns.heatmap(
        grid,
        ax=ax,
        cmap="vlag",
        center=NEUTRAL,
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "P(prefer target)", "shrink": 0.8},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Preference for the target language vs English")
    return fig


def _gradient(plt, palette, effect: pd.DataFrame):
    """Effect ordered by language resource level, one connected line per model."""
    targets = _ordered_targets(effect["target"].unique())
    x = range(len(targets))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for model in palette:
        ys = [_value(effect, model, t, "p_prefer_target") for t in targets]
        ax.plot(
            list(x),
            ys,
            "-o",
            color=palette[model],
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=short_model_name(model),
        )
    ax.axhline(NEUTRAL, color="0.4", lw=1.0, ls="--", zorder=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels([lang_label(t, multiline=True) for t in targets])
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(prefer target over English)")
    ax.set_title("Cross-linguistic gradient")
    if len(palette) > 1:
        ax.legend(title="model", loc="best")
    return fig


def _provenance(plt, palette, prov: pd.DataFrame):
    """Per-model P(prefer machine over human translation), with 95% CI, per sibling pair."""
    pairs = sorted(set(zip(prov["human"], prov["machine"])))
    models = list(palette)
    fig, ax = plt.subplots(figsize=(6.4, 1.2 + 0.5 * len(models) * max(len(pairs), 1)))
    yticks, ylabels, y = [], [], 0
    for human, machine in pairs:
        for model in models:
            row = prov[
                (prov["model"] == model)
                & (prov["human"] == human)
                & (prov["machine"] == machine)
            ]
            if row.empty:
                continue
            r = row.iloc[0]
            ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=palette[model], lw=1.6)
            ax.plot(
                r["p_prefer_target"],
                y,
                "o",
                color=palette[model],
                markersize=5.5,
                markeredgecolor="white",
                markeredgewidth=0.6,
            )
            yticks.append(y)
            ylabels.append(f"{short_model_name(model)}  ({human} vs {machine})")
            y += 1
    ax.axvline(NEUTRAL, color="0.4", lw=1.0, ls="--", zorder=0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(prefer the machine translation over the human one)")
    ax.set_title("Human vs machine translation")
    return fig


def _scale_ladder(plt, effect: pd.DataFrame):
    """Gemma size ladder: P(prefer target) vs parameter count, one line per language.

    Returns None when fewer than two Gemma sizes are present (nothing to compare)."""
    from matplotlib.ticker import ScalarFormatter

    sized = [(m, _gemma_size(m)) for m in models_in_table(effect)]
    sized = sorted((s, m) for m, s in sized if s is not None)
    if len(sized) < 2:
        return None
    sizes = [s for s, _ in sized]
    models = [m for _, m in sized]
    targets = _ordered_targets(effect["target"].unique())
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    lang_colors = _lang_colors(targets)
    for t in targets:
        ys = [_value(effect, m, t, "p_prefer_target") for m in models]
        ax.plot(
            sizes, ys, "-o", color=lang_colors[t], markersize=5, label=lang_label(t)
        )
    ax.axhline(NEUTRAL, color="0.4", lw=1.0, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Gemma 3 size (billion parameters, log scale)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(prefer target over English)")
    ax.set_title("Does the bias change with scale?")
    ax.legend(title="language", loc="best")
    return fig


def _controls(plt, controls: pd.DataFrame):
    """Per-model position and content baselines; both should sit on the 0.5 line."""
    models = models_in_table(controls)
    y = range(len(models))
    fig, ax = plt.subplots(figsize=(6.0, 1.2 + 0.45 * max(len(models), 1)))
    ax.plot(
        controls.set_index("model").loc[models, "position_first_rate"],
        list(y),
        "o",
        color="#3b6ea5",
        markersize=6,
        label="chose first-presented",
    )
    ax.plot(
        controls.set_index("model").loc[models, "content_pref_A_rate"],
        list(y),
        "s",
        color="#c44e52",
        markersize=6,
        label="chose content A",
    )
    ax.axvline(NEUTRAL, color="0.4", lw=1.0, ls="--", zorder=0)
    ax.set_yticks(list(y))
    ax.set_yticklabels([short_model_name(m) for m in models])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("rate (0.5 = no bias, counterbalancing held)")
    ax.set_title("Control diagnostics")
    ax.legend(loc="best")
    return fig


# --- small table helpers ---


def models_in_table(table: pd.DataFrame) -> list[str]:
    """Distinct models in a tidy result table, in stable sorted order."""
    return sorted(table["model"].unique())


def _value(table: pd.DataFrame, model: str, target: str, col: str):
    row = table[(table["model"] == model) & (table["target"] == target)]
    return float("nan") if row.empty else float(row.iloc[0][col])


def _lang_colors(targets: list[str]) -> dict[str, str]:
    import seaborn as sns

    palette = sns.color_palette("flare", n_colors=max(len(targets), 1))
    return {t: palette[i] for i, t in enumerate(targets)}


def _sanitize(name: str) -> str:
    """A LaTeX-command-safe CamelCase token (letters only) from a model/lang id."""
    parts = "".join(c if c.isalnum() else " " for c in name).split()
    return "".join(p.capitalize() for p in parts)


def _write_macros(path: Path, effect: pd.DataFrame) -> None:
    r"""Emit `\newcommand` macros for each (model, target) preference, e.g.
    `\UfGptFouroMiniAfr{23.9\%}`, so in-text numbers track the figures."""
    lines = [
        "% Auto-generated by `ufakazi figures`. Do not edit by hand.",
        "% P(prefer target-language testimony over English), percent.",
    ]
    seen: set[str] = set()
    for _, r in effect.iterrows():
        cmd = "Uf" + _sanitize(short_model_name(r["model"])) + _sanitize(r["target"])
        if cmd in seen:
            continue
        seen.add(cmd)
        lines.append(f"\\newcommand{{\\{cmd}}}{{{r['p_prefer_target'] * 100:.1f}\\%}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_figures(
    logs: str = "results/logs",
    outdir: str = "results/figures",
    *,
    reference: str = "en",
    n_boot: int = 2000,
    seed: int = 0,
) -> list[Path]:
    """Build every paper figure (PDF + PNG) and the tidy tables (CSV + macros.tex) from
    the eval logs in `logs`, written under `outdir`. Returns the paths written.

    Deterministic: the scenario bootstrap uses a fixed `seed`, so reruns are identical."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    from ufakazi.analysis.load import filter_latest_per_model

    df = filter_latest_per_model(load_trials(logs))
    if df.empty or not models_in(df):
        raise ValueError(f"No model results found in {logs!r}.")

    effect = language_effect_table(df, reference, n_boot=n_boot, seed=seed)
    prov = provenance_effect_table(df, n_boot=n_boot, seed=seed)
    controls = control_table(df)

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Tidy numbers first, so they exist even if a plot is skipped for sparse data.
    for name, table in (
        ("language_effect", effect),
        ("provenance_effect", prov),
        ("controls", controls),
    ):
        csv = out / f"{name}.csv"
        table.to_csv(csv, index=False)
        written.append(csv)
    macros = out / "macros.tex"
    _write_macros(macros, effect)
    written.append(macros)

    palette = _model_palette(sns, models_in_table(effect))
    builders = {
        "forest_language_effect": lambda: _forest(plt, palette, effect),
        "heatmap_model_language": lambda: _heatmap(plt, sns, effect),
        "gradient": lambda: _gradient(plt, palette, effect),
        "provenance": (lambda: _provenance(plt, palette, prov))
        if not prov.empty
        else None,
        "scale_ladder": lambda: _scale_ladder(plt, effect),
        "controls": lambda: _controls(plt, controls),
    }
    with plt.rc_context(PAPER_STYLE):
        for name, builder in builders.items():
            if builder is None:
                continue
            fig = builder()
            if fig is None:  # builder opted out (e.g. <2 Gemma sizes for the ladder)
                continue
            for ext in ("pdf", "png"):
                p = out / f"{name}.{ext}"
                fig.savefig(p)
                written.append(p)
            plt.close(fig)
    return written
