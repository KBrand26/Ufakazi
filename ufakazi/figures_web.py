"""Web-style paper figures for the blog post (docs/figures/).

A separate render pass from `figures.py`: same analysis tables, different design
language. The blog page (docs/index.html) encodes English as petrol and the
non-English testimony as ochre; these figures adopt the same semantic palette so
the reader decodes colour once and reuses it everywhere. Ramps are interpolated
in OKLab with monotone lightness per arm (checked numerically at render time),
and every cell annotation's ink is chosen by computed WCAG contrast.

Reads the final 9-model panel via `analysis.panel.load_panel` and, per figure,
writes a PNG (linkposts / OG asset) plus a light and a dark SVG, tokenised for
inlining; then swaps the SVG pairs into `docs/index.html` between its
`<!-- fig:NAME -->` markers and writes the tidy tables as CSV. Run:

    uv run python -m ufakazi.figures_web
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------- palette
# Page tokens (docs/index.html). Encoding hues (the ramp poles and the flag
# red) are the SAME in both themes: they carry meaning, and a legend must not
# lie. Surface-dependent colours (ink, muted, rules, the "no preference"
# neutral, and the surface the cell gaps expose) are per theme, and each
# theme's diverging ramp is anchored on ITS OWN neutral: a light midpoint on
# a light page, a dark midpoint on a dark page. Dark is a selected palette,
# not an inverted one.
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analysis.load import (
    language_effect_table,
    override_effect_table,
    provenance_effect_table,
)
from .analysis.panel import load_panel
from .analysis.rationales import rationale_appeal_table
from .analysis.verbosity import length_bias_table, length_ratio_table


@dataclass(frozen=True)
class Theme:
    name: str
    ink: str
    muted: str
    rule: str
    neutral: str  # diverging midpoint = "no preference"
    surface: str  # what shows through cell gaps / halo colour
    dim_ramp: float = 1.0  # multiply pole chroma/lightness reach on dark


LIGHT = Theme("light", "#16181D", "#636773", "#C6C6BD", "#EFEFEA", "#F5F5F1")
DARK = Theme("dark", "#E7E7E2", "#969BA6", "#3A3F49", "#20242B", "#111318")

OCHRE_POLE = "#A9702A"  # non-English pole (--other), both themes
FLAG = "#8C3A2E"  # override severity (--flag), both themes
EN_TOKEN = "#1D5C6B"  # --en, used to derive EN_POLE below

# Active theme; set by main() before each render pass. Module-level so the
# render helpers stay signature-compatible with the PNG path.
THEME: Theme = LIGHT

# ---------------------------------------------------------------- model panel
# Fixed entity order, reused by every figure: the two bias-robust models first,
# then the biased cohort (reasoning models, the non-reasoning baseline, and the
# Gemma scale ladder reading 4B -> 27B).
CLEAN = [
    ("openrouter/anthropic/claude-sonnet-4.6", "Claude Sonnet 4.6"),
    ("openrouter/openai/gpt-5.4", "GPT-5.4"),
]
BIASED = [
    ("openrouter/qwen/qwen3.7-plus", "Qwen 3.7 Plus"),
    ("openrouter/google/gemini-3.5-flash", "Gemini 3.5 Flash"),
    ("openrouter/x-ai/grok-4.3", "Grok 4.3"),
    ("openrouter/openai/gpt-4o-mini", "GPT-4o mini"),
    ("openrouter/google/gemma-3-4b-it", "Gemma 3 4B"),
    ("openrouter/google/gemma-3-12b-it", "Gemma 3 12B"),
    ("openrouter/google/gemma-3-27b-it", "Gemma 3 27B"),
]
LANGS = ["afr", "afr_mt", "zul_mt", "xho_mt"]
LANG_LABELS = [
    "Afrikaans\n(human)",
    "Afrikaans\n(machine)",
    "isiZulu\n(machine)",
    "isiXhosa\n(machine)",
]

# ---------------------------------------------------------------- OKLab math


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def _rgb_to_hex(rgb) -> str:
    return "#%02X%02X%02X" % tuple(round(max(0.0, min(1.0, c)) * 255) for c in rgb)


def _lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _unlin(c: float) -> float:
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def _rgb_to_oklab(rgb) -> tuple[float, float, float]:
    r, g, b = (_lin(c) for c in rgb)
    lm = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    lm, m, s = (c ** (1 / 3) for c in (lm, m, s))
    return (
        0.2104542553 * lm + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * lm - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * lm + 0.7827717662 * m - 0.8086757660 * s,
    )


def _oklab_to_rgb(lab) -> tuple[float, float, float]:
    L, a, b = lab
    lm = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    r = 4.0767416621 * lm - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * lm + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * lm - 0.7034186147 * m + 1.7076147010 * s
    return (_unlin(r), _unlin(g), _unlin(bb))


def _boost_chroma(hex_color: str, target_c: float) -> str:
    """Same OKLCH hue and lightness, chroma raised to `target_c` (sRGB-clipped)."""
    L, a, b = _rgb_to_oklab(_hex_to_rgb(hex_color))
    c = math.hypot(a, b)
    scale = target_c / c
    return _rgb_to_hex(_oklab_to_rgb((L, a * scale, b * scale)))


EN_POLE = _boost_chroma(EN_TOKEN, 0.115)


def _ramp(from_hex: str, to_hex: str, n: int) -> list[tuple[float, float, float]]:
    a = _rgb_to_oklab(_hex_to_rgb(from_hex))
    b = _rgb_to_oklab(_hex_to_rgb(to_hex))
    out = []
    for i in range(n):
        t = i / (n - 1)
        out.append(_oklab_to_rgb(tuple(x + (y - x) * t for x, y in zip(a, b))))
    return out


def _assert_monotone_l(colors, name: str) -> None:
    ls = [_rgb_to_oklab(c)[0] for c in colors]
    deltas = np.diff(ls)
    if not (np.all(deltas <= 1e-9) or np.all(deltas >= -1e-9)):
        raise AssertionError(f"{name}: ramp lightness is not monotone")


def _wcag_contrast(rgb_a, rgb_b) -> float:
    def lum(rgb):
        r, g, b = (_lin(c) for c in rgb)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    la, lb = lum(rgb_a), lum(rgb_b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


MIN_CELL_CONTRAST = 4.5  # WCAG AA for the numbers printed on heatmap cells


def _legible_pole(hex_color: str, contrast: float = MIN_CELL_CONTRAST) -> str:
    """The pole colour, with its OKLab lightness moved toward the theme's page
    ink contrast (lighter on a light page, darker on a dark page) just far
    enough that the page ink reads on it at `contrast`. Hue is kept; chroma is
    reduced only if the shifted colour leaves sRGB. Every heatmap cell then
    takes ONE ink (the page ink, no halos), because the fill ramp never gets
    darker (or lighter) than the ink can stand. Poles that already pass are
    returned unchanged."""
    ink = _hex_to_rgb(THEME.ink)
    L, a, b = _rgb_to_oklab(_hex_to_rgb(hex_color))
    ink_l = _rgb_to_oklab(ink)[0]
    step = 0.005 if ink_l < 0.5 else -0.005  # dark ink -> lighten; light ink -> darken
    for _ in range(200):
        rgb = _oklab_to_rgb((L, a, b))
        if max(rgb) > 1.0 or min(rgb) < 0.0:
            a, b = a * 0.97, b * 0.97  # pull chroma in until in gamut
            continue
        if _wcag_contrast(rgb, ink) >= contrast:
            return _rgb_to_hex(rgb)
        L += step
    raise AssertionError(f"could not make {hex_color} legible under {THEME.name} ink")


def _assert_ink_legible(colors, name: str) -> None:
    ink = _hex_to_rgb(THEME.ink)
    worst = min(_wcag_contrast(c, ink) for c in colors)
    if worst < MIN_CELL_CONTRAST - 0.05:
        raise AssertionError(
            f"{name}: page ink contrast {worst:.2f} < {MIN_CELL_CONTRAST}"
        )


# ---------------------------------------------------------------- output


def _tokenise_svg(svg: str, name: str) -> str:
    """Prepare a matplotlib SVG for inlining: namespace its ids so several
    figures (and both theme variants) can share one document, strip the
    hard-coded font family so the page's font applies, drop the fixed pt
    size so it scales with its container, and remove the XML prolog."""
    import re

    # namespace ids + their references (url(#id), href="#id")
    svg = re.sub(r'id="([^"]+)"', lambda m: f'id="{name}-{m.group(1)}"', svg)
    svg = re.sub(r"url\(#([^)]+)\)", lambda m: f"url(#{name}-{m.group(1)})", svg)
    svg = re.sub(r'href="#([^"]+)"', lambda m: f'href="#{name}-{m.group(1)}"', svg)
    # let the page control the font; matplotlib's own family list goes
    svg = re.sub(r"font-family:\s*[^;\"]+;?", "", svg)
    # responsive: drop fixed pt size, keep viewBox
    svg = re.sub(
        r'<svg([^>]*?)\swidth="[^"]+pt"\sheight="[^"]+pt"', r"<svg\1", svg, count=1
    )
    svg = svg.replace("<svg ", f'<svg class="fig fig-{THEME.name}" role="img" ', 1)
    # strip xml prolog / doctype / matplotlib's RDF metadata block for inlining
    svg = re.sub(r"<\?xml[^>]*\?>\s*", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg)
    svg = re.sub(r"\s*<metadata>.*?</metadata>", "", svg, count=1, flags=re.S)
    return svg.strip()


# Alt text per figure, applied when inlining (the <svg> gets role="img" +
# aria-label). Keys are the figure stems / the `<!-- fig:NAME -->` markers.
FIG_ALT = {
    "heatmap_model_language": (
        "Heatmap of how often each model chose the non-English account over the English "
        "one, by model and language."
    ),
    "provenance": (
        "Dot and 95% confidence interval per model for preferring machine over human "
        "Afrikaans; every interval hugs 0.5."
    ),
    "override_heatmap": (
        "Heatmap of the share of saturated scenarios in which the model flipped its "
        "choice."
    ),
    "length": (
        "Dot strip per language of each translated testimony's length relative to its "
        "English original, with the mean marked; Afrikaans runs about 1.1 times longer, "
        "isiZulu and isiXhosa sit on the 1x line."
    ),
    "rationale_appeals": (
        "Dumbbell per model: share of rationales appealing to language when the model "
        "chose the English testimony versus the other, with the gap annotated."
    ),
}


def inline_figures(html_path: Path, fig_dir: Path) -> list[str]:
    """Replace each `<!-- fig:NAME --> ... <!-- /fig:NAME -->` block in `html_path`
    with the light + dark SVG pair from `fig_dir` (CSS on the page switches them by
    theme). Markers without a rendered figure, or figures without a marker, are left
    alone and reported. Returns the names inlined."""
    import re

    html = html_path.read_text()
    done = []
    for name, alt in FIG_ALT.items():
        light, dark = fig_dir / f"{name}.svg", fig_dir / f"{name}.dark.svg"
        marker = re.compile(rf"(<!-- fig:{name} -->)(.*?)(<!-- /fig:{name} -->)", re.S)
        if not marker.search(html):
            print(f"  no <!-- fig:{name} --> marker in {html_path.name}; skipped")
            continue
        if not (light.exists() and dark.exists()):
            print(f"  {name}: svg pair not rendered; skipped")
            continue
        svgs = []
        for path in (light, dark):
            svg = path.read_text()
            svg = svg.replace('role="img" ', f'role="img" aria-label="{alt}" ', 1)
            svgs.append(svg)
        block = "\n".join(svgs)
        html = marker.sub(lambda m: f"{m.group(1)}\n{block}\n    {m.group(3)}", html)
        done.append(name)
    html_path.write_text(html)
    return done


def _save(fig, out: Path) -> None:
    """Light pass: write `out` (.png, the linkpost/OG asset, on a white plate)
    plus `<stem>.svg`. Dark pass: write `<stem>.dark.svg` only. SVGs are
    transparent and self-consistent for their theme, ready to inline."""
    import io

    if THEME is LIGHT:
        fig.savefig(out, facecolor="white", transparent=False)
        svg_path = out.with_suffix(".svg")
    else:
        svg_path = out.with_suffix(f".{THEME.name}.svg")
    buf = io.StringIO()
    fig.savefig(buf, format="svg", transparent=True)
    svg_path.write_text(
        _tokenise_svg(buf.getvalue(), f"{out.stem.replace('_', '-')}-{THEME.name}")
    )


# ---------------------------------------------------------------- rendering


def _style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "monospace",
            "font.monospace": ["Menlo", "Consolas", "DejaVu Sans Mono"],
            "font.size": 10,
            "text.color": THEME.ink,
            "axes.edgecolor": THEME.rule,
            "axes.labelcolor": THEME.ink,
            "xtick.color": THEME.muted,
            "ytick.color": THEME.ink,
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.facecolor": "none",
            "savefig.transparent": True,
            "savefig.dpi": 200,
            "svg.fonttype": "none",  # keep text as <text>, inherits page font
        }
    )


def _diverging_cmap():
    from matplotlib.colors import ListedColormap

    left = _ramp(_legible_pole(EN_POLE), THEME.neutral, 256)
    right = _ramp(THEME.neutral, _legible_pole(OCHRE_POLE), 256)
    _assert_monotone_l(left, "diverging left arm")
    _assert_monotone_l(right, "diverging right arm")
    _assert_ink_legible(left + right, "diverging ramp")
    return ListedColormap(left + right, name="petrol_ochre")


def _sequential_cmap():
    from matplotlib.colors import ListedColormap

    colors = _ramp(THEME.neutral, _legible_pole(FLAG), 256)
    _assert_monotone_l(colors, "sequential flag ramp")
    _assert_ink_legible(colors, "sequential flag ramp")
    return ListedColormap(colors, name="flag")


def _grouped_heatmap(
    values: np.ndarray,
    annotations: list[list[str]],
    bold: np.ndarray,
    cmap,
    cbar_label: str,
    cbar_ends: tuple[str, str],
    out: Path,
) -> None:
    """9x4 heatmap split into the bias-robust pair and the biased cohort, one
    shared colour scale, 2px white cell gaps, contrast-picked annotation ink."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    norm = Normalize(0.0, 1.0)
    n_clean, n_biased = len(CLEAN), len(BIASED)
    fig = plt.figure(figsize=(6.7, 6.55), constrained_layout=True)
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[n_clean, n_biased, 0.32],
        hspace=0.06,
    )
    ax_clean = fig.add_subplot(gs[0])
    ax_biased = fig.add_subplot(gs[1])
    ax_cbar = fig.add_subplot(gs[2])

    groups = [
        (ax_clean, 0, n_clean, "NO SIGNIFICANT BIAS"),
        (ax_biased, n_clean, n_clean + n_biased, "BIASED TOWARDS ENGLISH"),
    ]
    labels = [label for _, label in CLEAN + BIASED]
    for ax, lo, hi, caption in groups:
        block = values[lo:hi]
        ax.pcolormesh(
            block, cmap=cmap, norm=norm, edgecolors=THEME.surface, linewidth=2.0
        )
        ax.invert_yaxis()
        ax.set_aspect("auto")
        ax.set_yticks(np.arange(hi - lo) + 0.5, labels[lo:hi], fontsize=9.5)
        ax.set_xticks([])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(caption, loc="left", fontsize=7.5, color=THEME.muted, pad=6)
        for r in range(hi - lo):
            for c in range(values.shape[1]):
                v = values[lo + r, c]
                if np.isnan(v):
                    continue
                # One ink for every cell: the ramp is built so the page ink
                # reads on any fill (see _legible_pole), so nothing flips to
                # white or needs a halo, and bold stays the only emphasis.
                ax.text(
                    c + 0.5,
                    r + 0.5,
                    annotations[lo + r][c],
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=THEME.ink,
                    fontweight="bold" if bold[lo + r, c] else "normal",
                )
    ax_biased.set_xticks(np.arange(len(LANG_LABELS)) + 0.5, LANG_LABELS, fontsize=9.5)

    from matplotlib.cm import ScalarMappable

    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=ax_cbar,
        orientation="horizontal",
    )
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    cbar.ax.spines["outline"].set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=8.5, colors=THEME.muted)
    cbar.set_label(cbar_label, fontsize=9, color=THEME.ink, labelpad=6)
    cbar.ax.text(
        0,
        1.35,
        cbar_ends[0],
        transform=cbar.ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color=THEME.muted,
    )
    cbar.ax.text(
        1,
        1.35,
        cbar_ends[1],
        transform=cbar.ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=THEME.muted,
    )
    _save(fig, out)
    plt.close(fig)


def _panel_matrix(table, value_col: str, langs=LANGS):
    """(models x langs) value and significance matrices in fixed panel order."""
    order = [m for m, _ in CLEAN + BIASED]
    vals = np.full((len(order), len(langs)), np.nan)
    sig = np.zeros((len(order), len(langs)), dtype=bool)
    for i, model in enumerate(order):
        for j, lang in enumerate(langs):
            row = table[(table["model"] == model) & (table["target"] == lang)]
            if len(row):
                vals[i, j] = row.iloc[0][value_col]
                if "significant" in row.columns:
                    sig[i, j] = bool(row.iloc[0]["significant"])
    return vals, sig


def render_language_heatmap(lang_table, out: Path) -> None:
    vals, sig = _panel_matrix(lang_table, "p_prefer_target")
    annotations = [["" if np.isnan(v) else f"{v:.0%}" for v in row] for row in vals]
    _grouped_heatmap(
        vals,
        annotations,
        sig,
        _diverging_cmap(),
        "How often the model chose the non-English account\n(balanced scenarios)",
        ("← always chose English", "always chose the other language →"),
        out,
    )


def render_override_heatmap(override_table, out: Path) -> None:
    vals, _ = _panel_matrix(override_table, "flip_fraction")
    annotations = [["" if np.isnan(v) else f"{v:.0%}" for v in row] for row in vals]
    _grouped_heatmap(
        vals,
        annotations,
        np.zeros_like(vals, dtype=bool),
        _sequential_cmap(),
        "How often the language change flipped the choice\n(saturated scenarios)",
        ("← never flipped", "always flipped →"),
        out,
    )


def render_provenance(prov_table, out: Path) -> None:
    """Dot + 95% CI per model: P(prefer machine over human Afrikaans). The
    full 0-1 axis is deliberate; the cluster hugging 0.5 is the finding."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    # Same diverging scale as Fig 1, so the plot speaks the page's colour
    # language: petrol = prefers the human translation, ochre = prefers the
    # machine one. Marks are coloured by their own value on that scale, so
    # they render as barely-tinted neutrals: colour scarcely appears because
    # the effect scarcely exists.
    cmap = _diverging_cmap()
    norm = Normalize(0.0, 1.0)
    order = CLEAN + BIASED
    fig, ax = plt.subplots(figsize=(8.0, 3.9), constrained_layout=True)
    n = len(order)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, n - 0.3)
    # Two-sided gradient wash across the whole axis: strongest at the poles,
    # fading to nothing at 0.5. It anchors the palette without shouting.
    xs = np.linspace(0, 1, 512)
    wash = cmap(norm(xs))[None, :, :3]
    ax.imshow(
        wash,
        extent=(0, 1, -0.7, n - 0.3),
        aspect="auto",
        alpha=0.55,
        zorder=0,
        interpolation="bilinear",
    )
    dagger_rows = []
    for i, (model, label) in enumerate(order):
        row = prov_table[prov_table["model"] == model]
        if not len(row):
            continue
        r = row.iloc[0]
        y = n - 1 - i
        p = float(r["p_prefer_target"])
        # Fill = own value on the scale; edge = its pole hue, so even the
        # near-neutral estimates carry a readable side.
        pole = EN_POLE if p < 0.5 else OCHRE_POLE
        ax.hlines(y, r["ci_lo"], r["ci_hi"], color=pole, linewidth=2.6, zorder=2)
        ax.plot(
            p,
            y,
            "o",
            markersize=9,
            markerfacecolor=cmap(norm(p))[:3],
            markeredgecolor=pole,
            markeredgewidth=1.6,
            zorder=3,
        )
        if bool(r["significant"]):
            dagger_rows.append((label, r))
    labels = [
        label + (" †" if any(label == d[0] for d in dagger_rows) else "")
        for _, label in order
    ]
    ax.set_yticks([n - 1 - i for i in range(n)], labels, fontsize=9.5)
    ax.axvline(0.5, color=THEME.ink, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.axhline(n - len(CLEAN) - 0.5, color=THEME.rule, linewidth=1, zorder=1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(length=0, labelsize=8.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    # Direction cues sit just under the tick labels; the xlabel (with the
    # dagger footnote as its second line) clears them via labelpad, so the
    # whole bottom stack stays inside constrained_layout's accounting and
    # nothing gets clipped.
    ax.text(
        0.01,
        -0.095,
        "← preferred the human translation",
        transform=ax.transAxes,
        fontsize=7.5,
        color=THEME.muted,
        ha="left",
        va="top",
    )
    ax.text(
        0.99,
        -0.095,
        "preferred the machine translation →",
        transform=ax.transAxes,
        fontsize=7.5,
        color=THEME.muted,
        ha="right",
        va="top",
    )
    xlabel = "How often the model chose the machine translation over the human one"
    foot = "bars are 95% intervals"
    if dagger_rows:
        notes = "; ".join(
            f"{lab.strip()}: {r['p_prefer_target']:.0%} ({r['ci_lo']:.0%} to {r['ci_hi']:.0%})"
            for lab, r in dagger_rows
        )
        # Own line: the page renders SVG text in its own monospace font, which can
        # run wider than matplotlib's, so long label lines need real slack.
        foot += f"\n† interval excludes 50%: {notes}"
    xlabel += "\n" + foot
    ax.set_xlabel(xlabel, fontsize=9, labelpad=32)
    _save(fig, out)
    plt.close(fig)


def render_rationale_appeals(appeal_table, out: Path) -> None:
    """Dumbbell per model: share of rationales that appeal to language when the
    model chose the English side (petrol) vs when it chose the other side
    (ochre). The connector length is the gap; a dagger marks a gap whose
    scenario-bootstrap CI excludes zero. Same fixed model order as the other
    figures so the bias-robust rows read first."""
    import matplotlib.pyplot as plt

    order = CLEAN + BIASED
    n = len(order)
    fig, ax = plt.subplots(figsize=(8.0, 3.9), constrained_layout=True)
    ax.set_xlim(0, 0.75)
    ax.set_ylim(-0.7, n - 0.3)
    labels = []
    for i, (model, label) in enumerate(order):
        row = appeal_table[appeal_table["model"] == model]
        if not len(row):
            labels.append(label)
            continue
        r = row.iloc[0]
        y = n - 1 - i
        a, b = float(r["appeal_rate_ref"]), float(r["appeal_rate_other"])
        ax.hlines(y, min(a, b), max(a, b), color=THEME.rule, linewidth=2.4, zorder=1)
        ax.plot(b, y, "o", ms=9, mfc=OCHRE_POLE, mec=OCHRE_POLE, zorder=3)
        ax.plot(a, y, "o", ms=9, mfc=EN_POLE, mec=EN_POLE, zorder=3)
        sig = bool(r["appeal_gap_significant"])
        ax.text(
            0.735,
            y,
            f"{r['appeal_gap'] * 100:+.0f} pts" + (" †" if sig else ""),
            ha="right",
            va="center",
            fontsize=8.5,
            color=THEME.ink,
            fontweight="bold" if sig else "normal",
            zorder=4,
        )
        labels.append(label)
    ax.set_yticks([n - 1 - i for i in range(n)], labels, fontsize=9.5)
    ax.axhline(n - len(CLEAN) - 0.5, color=THEME.rule, linewidth=1, zorder=1)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_xticklabels(["0%", "20%", "40%", "60%"])
    ax.tick_params(length=0, labelsize=8.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.text(
        0.735,
        n - 0.3,
        "gap",
        ha="right",
        va="bottom",
        fontsize=8,
        color=THEME.muted,
    )
    # Legend: two series, so a legend box is required; direct-label the first
    # row too so identity is never colour-alone.
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=8,
            color=EN_POLE,
            label="when it chose the English account",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=8,
            color=OCHRE_POLE,
            label="when it chose the other account",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.42, 1.0),
        frameon=False,
        fontsize=8.5,
        handletextpad=0.4,
        labelcolor=THEME.ink,
    )
    ax.set_xlabel(
        "How often the model's reason cited the language or the translation"
        "\n† difference is statistically significant (95% interval excludes zero)",
        fontsize=9,
        labelpad=10,
    )
    _save(fig, out)
    plt.close(fig)


def render_length(ratio_table, out: Path) -> None:
    """Dot strip per language: every translated testimony's length relative to its
    English original (characters), with the per-language mean marked. The 1x line
    is the English reference (petrol); the translations are ochre, as everywhere
    else on the page. Rows keep the heatmap's language order, which is also the
    order of increasing bias, so reading down the strip the text gets shorter
    while (from Fig 1) the models get harsher: the length story runs the wrong way."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    n = len(LANGS)
    fig, ax = plt.subplots(figsize=(8.0, 3.2), constrained_layout=True)
    ax.set_xlim(0.6, 1.5)
    ax.set_ylim(-0.7, n - 0.3)
    labels = []
    for i, (lang, label) in enumerate(zip(LANGS, LANG_LABELS)):
        y = n - 1 - i
        xs = ratio_table.loc[ratio_table["target"] == lang, "ratio"].to_numpy()
        labels.append(label.replace("\n", " "))
        if not len(xs):
            continue
        # One faint dot per testimony, jittered within the row; the mean is the
        # only solid mark, so the eye reads cloud + centre, not a scatter.
        jitter = rng.uniform(-0.18, 0.18, size=len(xs))
        ax.plot(
            xs,
            y + jitter,
            "o",
            ms=3.4,
            mfc=OCHRE_POLE,
            mec="none",
            alpha=0.35,
            zorder=2,
        )
        m = float(xs.mean())
        ax.plot(m, y, "o", ms=9.5, mfc=OCHRE_POLE, mec=THEME.ink, mew=1.2, zorder=4)
        ax.text(
            m,
            y + 0.36,
            f"{m:.2f}×",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=THEME.ink,
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.12", fc=THEME.surface, ec="none"),
        )
    ax.axvline(1.0, color=EN_POLE, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.text(
        1.0,
        n - 0.32,
        "English original ",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=EN_POLE,
    )
    ax.set_yticks([n - 1 - i for i in range(n)], labels, fontsize=9.5)
    ax.set_xticks([0.75, 1.0, 1.25, 1.5])
    ax.set_xticklabels(["0.75×", "1×", "1.25×", "1.5×"])
    ax.tick_params(length=0, labelsize=8.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.text(
        0.01,
        -0.12,
        "← shorter than the English",
        transform=ax.transAxes,
        fontsize=7.5,
        color=THEME.muted,
        ha="left",
        va="top",
    )
    ax.text(
        0.99,
        -0.12,
        "longer than the English →",
        transform=ax.transAxes,
        fontsize=7.5,
        color=THEME.muted,
        ha="right",
        va="top",
    )
    ax.set_xlabel(
        "Testimony length relative to the English original (characters)"
        "\nfaint dots are the 40 testimonies per language  ·  solid dot is their mean",
        fontsize=9,
        labelpad=30,
    )
    _save(fig, out)
    plt.close(fig)


# ---------------------------------------------------------------- entrypoint


def main(out_dir: str = "docs/figures", page: str | None = "docs/index.html") -> None:
    """Render every blog figure (light + dark), write the tidy tables, and inline
    the SVG pairs into `page` (skipped if the file is absent)."""
    _style()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Pooling recipe lives in analysis/panel.py (Gemini spans two evals; 4o-mini's
    # 10-epoch run sits in results/logs).
    df = load_panel()
    print("panel:")
    for model, sub in df.groupby("model"):
        print(f"  {model}: {len(sub)} trials")

    lang = language_effect_table(df)
    prov = provenance_effect_table(df)
    over = override_effect_table(df)
    appeals = rationale_appeal_table(df)
    lengths = length_ratio_table()
    # The prose companion to the length figure (P(chose the longer account) inside
    # same-language controls, per model); tabled so the in-text range is checkable.
    length_bias = length_bias_table(df)
    # Tidy tables alongside the figures, so in-text numbers can be checked
    # against exactly what was plotted.
    for name, table in (
        ("language_effect", lang),
        ("provenance_effect", prov),
        ("override_effect", over),
        ("rationale_appeals", appeals),
        ("length_ratio", lengths),
        ("length_bias", length_bias),
    ):
        table.to_csv(out / f"{name}.csv", index=False)

    global THEME
    for theme in (LIGHT, DARK):
        THEME = theme
        _style()
        render_language_heatmap(lang, out / "heatmap_model_language.png")
        render_provenance(prov, out / "provenance.png")
        render_override_heatmap(over, out / "override_heatmap.png")
        render_rationale_appeals(appeals, out / "rationale_appeals.png")
        render_length(lengths, out / "length.png")
    THEME = LIGHT
    print(f"EN ramp pole (chroma-boosted from {EN_TOKEN}): {EN_POLE}")
    print(f"wrote 5 figures x (png, svg, dark.svg) + 6 csv tables -> {out}")
    if page and Path(page).exists():
        # Page assembly (figure inlining + the generated Sources list) lives in
        # ufakazi.site; run it here so one command refreshes the whole page.
        from .site import build

        build(Path(page), out)


if __name__ == "__main__":
    main()
