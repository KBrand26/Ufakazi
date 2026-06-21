"""Render the latest study run into a self-contained HTML results artefact.

Reads the most recent `.eval` log through the analysis layer (so the numbers always match
`ufakazi analyze`) and writes `artefacts/results.html`: a single file with inline CSS and
SVG, no external dependencies, that opens offline anywhere.

    uv run python artefacts/generate_report.py
"""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
from inspect_ai.analysis import evals_df

from ufakazi.analysis.load import (
    control_baselines,
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

LANG_NAME = {
    "en": "English",
    "afr": "Afrikaans",
    "afr_mt": "Machine Afrikaans",
    "zul_mt": "isiZulu",
    "xho_mt": "isiXhosa",
}

# (display name, translation provenance) for the cross-language ladder. Afrikaans is the
# one human-verified arm; the rest are machine-translated, which the caption flags.
LADDER_LABELS = {
    "afr": ("Afrikaans", "human"),
    "afr_mt": ("Afrikaans", "machine"),
    "zul_mt": ("isiZulu", "machine"),
    "xho_mt": ("isiXhosa", "machine"),
}

HERE = Path(__file__).parent
OUT = HERE / "results.html"
LOGO = HERE / "ufakazi-logo.png"
LOG_DIR = "results/logs"

# Brand palette, taken from the Ufakazi eye logo / README badges. Teal encodes
# "English-leaning" and amber "Afrikaans-leaning" consistently across every chart.
TEAL = "#5BA893"
TEAL_DARK = "#2f6b5e"
TEAL_LIGHT = "#9bd3c5"
AMBER = "#E0A24A"
AMBER_DARK = "#b97e2e"
CREAM = "#ece6d6"


def _logo_data_uri() -> str:
    """Base64 data URI for the logo, so the HTML stays a single self-contained file."""
    if not LOGO.exists():
        return ""
    encoded = base64.b64encode(LOGO.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _latest_eval_meta(df_trials: pd.DataFrame) -> dict:
    """Model + creation date for the run the trials came from."""
    if "eval_id" not in df_trials.columns or df_trials.empty:
        return {"model": "the evaluated model", "created": ""}
    eval_id = df_trials["eval_id"].iloc[0]
    evals = evals_df(LOG_DIR)
    row = evals[evals["eval_id"] == eval_id]
    if row.empty:
        return {"model": "the evaluated model", "created": ""}
    created = str(row["created"].iloc[0])
    return {"model": str(row["model"].iloc[0]), "created": created[:10]}


def _main_effect_svg(p: float, lo: float, hi: float, cont: float | None) -> str:
    """Horizontal 0..1 scale: point estimate (binary), bootstrap CI, continuous marker,
    and the 0.5 null line. Labels are kept off the crowded centre to avoid collisions; the
    two close markers are disambiguated by the legend rendered beneath the SVG."""
    w, x0, x1, y = 760, 64, 696, 104
    plot = x1 - x0

    def x(v: float) -> float:
        return x0 + v * plot

    ticks = ""
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        ticks += (
            f'<line x1="{x(v):.1f}" y1="{y}" x2="{x(v):.1f}" y2="{y + 6}" '
            f'stroke="#cbd5e1" stroke-width="1"/>'
            f'<text x="{x(v):.1f}" y="{y + 23}" text-anchor="middle" '
            f'class="svg-tick">{v:.2f}</text>'
        )
    # Continuous marker (hollow amber ring), no text label: explained in the legend below.
    cont_marker = ""
    if cont is not None:
        cont_marker = (
            f'<circle cx="{x(cont):.1f}" cy="{y}" r="6.5" fill="white" '
            f'stroke="{AMBER_DARK}" stroke-width="2.5"/>'
        )
    return f"""<svg viewBox="0 0 {w} 150" class="effect-svg" role="img"
      aria-label="Probability the model preferred the Afrikaans testimony">
  <rect x="{x0}" y="{y - 11}" width="{(x(0.5) - x0):.1f}" height="22" fill="{TEAL}" opacity="0.09"/>
  <rect x="{x(0.5):.1f}" y="{y - 11}" width="{(x1 - x(0.5)):.1f}" height="22" fill="{AMBER}" opacity="0.11"/>
  <line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#94a3b8" stroke-width="1.5"/>
  {ticks}
  <text x="{x0}" y="{y - 50}" text-anchor="start" class="svg-end" fill="{TEAL_DARK}">&#9664; prefers English</text>
  <text x="{x1}" y="{y - 50}" text-anchor="end" class="svg-end" fill="{AMBER_DARK}">prefers Afrikaans &#9654;</text>
  <line x1="{x(0.5):.1f}" y1="{y - 38}" x2="{x(0.5):.1f}" y2="{y + 10}"
        stroke="#64748b" stroke-width="1.5" stroke-dasharray="4 3"/>
  <text x="{x(0.5):.1f}" y="{y - 26}" text-anchor="middle" class="svg-null">no preference (0.50)</text>
  <rect x="{x(lo):.1f}" y="{y - 8}" width="{(x(hi) - x(lo)):.1f}" height="16"
        rx="3" fill="{TEAL}" opacity="0.30"/>
  <line x1="{x(lo):.1f}" y1="{y - 9}" x2="{x(lo):.1f}" y2="{y + 9}" stroke="{TEAL_DARK}" stroke-width="2"/>
  <line x1="{x(hi):.1f}" y1="{y - 9}" x2="{x(hi):.1f}" y2="{y + 9}" stroke="{TEAL_DARK}" stroke-width="2"/>
  {cont_marker}
  <circle cx="{x(p):.1f}" cy="{y}" r="9" fill="{TEAL}" stroke="white" stroke-width="2.5"/>
  <text x="{x(p):.1f}" y="{y - 24}" text-anchor="middle" class="svg-point">{p:.3f}</text>
</svg>"""


def _scenario_bars(effects: pd.DataFrame) -> str:
    rows = ""
    for _, r in effects.iterrows():
        v = float(r["language_effect"])
        pct = min(abs(v), 1.0) * 50.0
        if v < 0:
            bar = f'<div class="scn-bar neg" style="right:50%;width:{pct:.1f}%"></div>'
        else:
            bar = f'<div class="scn-bar pos" style="left:50%;width:{pct:.1f}%"></div>'
        rows += f"""<div class="scn-row">
        <div class="scn-name">{r["scenario_id"]}</div>
        <div class="scn-track"><div class="scn-center"></div>{bar}</div>
        <div class="scn-val">{v:+.2f}</div>
      </div>"""
    return rows


def _language_ladder_svg(rows: list[dict]) -> str:
    """Forest plot: P(prefer each language over English) with bootstrap CIs on one shared
    0..1 axis. Every point left of 0.50 means English was preferred; the vertical spread
    between the points is the cross-linguistic gradient that is the card's whole message."""
    w, x0, x1 = 760, 196, 712
    plot = x1 - x0
    top, step = 46, 44
    axis_y = top + len(rows) * step + 2
    height = axis_y + 36

    def x(v: float) -> float:
        return x0 + v * plot

    ticks = ""
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        ticks += (
            f'<line x1="{x(v):.1f}" y1="{top - 8}" x2="{x(v):.1f}" y2="{axis_y}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
            f'<text x="{x(v):.1f}" y="{axis_y + 18:.0f}" text-anchor="middle" '
            f'class="svg-tick">{v:.2f}</text>'
        )
    body = ""
    for i, r in enumerate(rows):
        cy = top + i * step + step / 2 - 4
        p, lo, hi = r["p"], r["lo"], r["hi"]
        body += (
            f'<text x="{x0 - 16:.0f}" y="{cy - 1:.0f}" text-anchor="end" '
            f'class="lad-lab">{r["name"]}</text>'
            f'<text x="{x0 - 16:.0f}" y="{cy + 14:.0f}" text-anchor="end" '
            f'class="lad-prov">{r["prov"]}</text>'
            f'<line x1="{x(lo):.1f}" y1="{cy:.1f}" x2="{x(hi):.1f}" y2="{cy:.1f}" '
            f'stroke="{TEAL}" stroke-width="6" stroke-linecap="round" opacity="0.35"/>'
            f'<circle cx="{x(p):.1f}" cy="{cy:.1f}" r="7" fill="{TEAL}" '
            f'stroke="white" stroke-width="2"/>'
            f'<text x="{x(p):.1f}" y="{cy - 13:.0f}" text-anchor="middle" '
            f'class="lad-val">{p:.3f}</text>'
        )
    return f"""<svg viewBox="0 0 {w} {height}" class="effect-svg" role="img"
      aria-label="Preference for each language over English, with confidence intervals">
  <rect x="{x0}" y="{top - 8}" width="{(x(0.5) - x0):.1f}" height="{(axis_y - top + 8):.1f}"
        fill="{TEAL}" opacity="0.05"/>
  {ticks}
  <line x1="{x(0.5):.1f}" y1="{top - 26}" x2="{x(0.5):.1f}" y2="{axis_y}"
        stroke="#64748b" stroke-width="1.5" stroke-dasharray="4 3"/>
  <text x="{x(0.5):.1f}" y="{top - 30:.0f}" text-anchor="middle" class="svg-null">English preferred &#9664; | &#9654; other language preferred (0.50)</text>
  {body}
</svg>"""


def _language_gradient_card(reports: list[dict], reference: str) -> str:
    """The cross-linguistic result: how the language penalty varies across all targets,
    not just Afrikaans. Renders only when at least two non-reference languages were run."""
    targets = [r for r in reports if not pd.isna(r["p_prefer_target"])]
    if len(targets) < 2:
        return ""
    ordered = sorted(targets, key=lambda r: r["p_prefer_target"], reverse=True)
    rows = []
    for r in ordered:
        name, prov = LADDER_LABELS.get(r["target"], (r["target"], ""))
        rows.append(
            {
                "name": name,
                "prov": prov,
                "p": r["p_prefer_target"],
                "lo": r["ci95"][0],
                "hi": r["ci95"][1],
            }
        )
    ref_name = LANG_NAME.get(reference, reference)
    top_r, bot_r = rows[0], rows[-1]
    top_lab = f"{top_r['name']} ({top_r['prov']})"
    bot_lab = f"{bot_r['name']} ({bot_r['prov']})"
    machine_only = [
        r["name"]
        for r in rows
        if r["prov"] == "machine" and r["name"] not in ("Afrikaans",)
    ]
    machine_phrase = (
        " and ".join(dict.fromkeys(machine_only))
        if machine_only
        else "the machine-translated languages"
    )
    conclusion = (
        f"Every language is judged less credible than {ref_name}, but the penalty is "
        f"<strong>graded, not flat</strong>. {top_lab} keeps the most credibility "
        f"(<b>{top_r['p']:.2f}</b>), while {bot_lab} is chosen over {ref_name} only "
        f"<b>{bot_r['p'] * 100:.0f}%</b> of the time (<b>{bot_r['p']:.2f}</b>) &mdash; roughly a "
        f"third of the Afrikaans rate. The content is identical across all renderings, so this "
        f"ordering is about the <strong>language</strong> itself, and it tracks how much text the "
        f"model has likely seen in each: a resource- and familiarity-driven bias. One caution: "
        f"unlike Afrikaans, which a native speaker verified and where human and machine versions "
        f"behaved alike (below), {machine_phrase} are machine-translated and unverified, so part of "
        f"their steeper penalty could be translation quality rather than language identity."
    )
    return f"""
    <div class="card">
      <h2>Does the penalty hold across languages?</h2>
      <p class="sub">The same forced choice, now with four languages set against the {ref_name}
        source. Each point is P(prefer that language over {ref_name}); the bar is its 95%
        scenario-bootstrap interval. Further left means a stronger pull toward {ref_name}.</p>
      {_language_ladder_svg(rows)}
      <p class="caption">{conclusion}</p>
    </div>"""


def _verbosity_facts(df: pd.DataFrame, target: str, reference: str) -> dict | None:
    """Flatten the three verbosity analyses into the primitives the card needs.

    Returns None when there are no content-matched target/reference pairs to measure."""
    lens = length_summary(target, reference)
    if not lens:
        return None
    bias = length_bias_controls(df)
    corr = effect_vs_length(df, target, reference)
    return {
        "target": target,
        "reference": reference,
        "ratio": lens["mean_ratio_target_over_reference"],
        "mean_ref": lens[f"mean_len_{reference}"],
        "mean_tgt": lens[f"mean_len_{target}"],
        "p_longer": bias["p_chose_longer"],
        "ci": bias["ci95"],
        "sig": bias["significant"],
        "r": corr["pearson_r"],
    }


def _verbosity_card(v: dict | None, p_prefer_target: float) -> str:
    """Rebuttal card to the 'isn't it just length?' confound.

    `v` carries the length facts; `p_prefer_target` is the headline language preference
    (P(prefer target) < 0.5 means the model prefers the reference language). The conclusion
    is derived from the numbers, so it stays correct if a re-run flips a direction."""
    if not v:
        return ""
    tgt, ref = (
        LANG_NAME.get(v["target"], v["target"]),
        LANG_NAME.get(v["reference"], v["reference"]),
    )
    longer_w = 100.0
    shorter_w = max(
        8.0,
        min(v["mean_ref"], v["mean_tgt"]) / max(v["mean_ref"], v["mean_tgt"]) * 100.0,
    )
    tgt_longer = v["mean_tgt"] >= v["mean_ref"]
    tgt_w, ref_w = (longer_w, shorter_w) if tgt_longer else (shorter_w, longer_w)
    lo, hi = v["ci"]

    # Which language does a pure length preference push toward, and does that oppose the
    # measured language preference? If it opposes, length cannot explain the effect.
    prefers_longer = v["p_longer"] > 0.5
    length_favours_target = prefers_longer == (v["ratio"] > 1)
    prefers_target_lang = p_prefer_target > 0.5
    opposes = length_favours_target != prefers_target_lang
    lean = "longer" if prefers_longer else "shorter"
    longer_lang = tgt if tgt_longer else ref
    sig_phrase = (
        "a statistically significant bias"
        if v["sig"]
        else "not statistically significant"
    )
    if opposes:
        conclusion = (
            f"The premise holds: {tgt} testimonies do run <b>{v['ratio']:.2f}&times;</b> "
            f"longer. But the model's own length preference points the other way. Within "
            f"same-language controls it leans toward the <b>{lean}</b> testimony "
            f"(<b>{v['p_longer']:.2f}</b>, {sig_phrase}), so a pure length preference would "
            f"favour the longer {longer_lang} text &mdash; the opposite of the {ref} "
            f"preference we measured. Across scenarios, the more {tgt} outruns {ref} in "
            f"length, the slightly more {tgt} is chosen (r = <b>{v['r']:+.2f}</b>), again the "
            f"wrong sign for a 'shorter is more credible' story. Verbosity does not explain "
            f"the effect; if anything it masks it."
        )
    else:
        conclusion = (
            f"{tgt} testimonies run <b>{v['ratio']:.2f}&times;</b> longer, and within "
            f"same-language controls the model leans toward the <b>{lean}</b> testimony "
            f"(<b>{v['p_longer']:.2f}</b>, {sig_phrase}). That points the same way as the "
            f"measured language preference, so length is a <b>partial confound</b> here and "
            f"the language effect cannot be cleanly separated from verbosity on this run."
        )
    return f"""
    <div class="card">
      <h2>Is it just verbosity?</h2>
      <p class="sub">Translation tends to lengthen text, so a model that simply favoured
        shorter testimonies would look like it favoured English. Two checks test that.</p>
      <div class="lenbars">
        <div class="lenbar-row"><span class="lenbar-lab">{ref}</span>
          <div class="lenbar"><div class="lenbar-fill en" style="width:{ref_w:.1f}%"></div></div>
          <span class="lenbar-num">{v["mean_ref"]:.0f} chars</span></div>
        <div class="lenbar-row"><span class="lenbar-lab">{tgt}</span>
          <div class="lenbar"><div class="lenbar-fill afr" style="width:{tgt_w:.1f}%"></div></div>
          <span class="lenbar-num">{v["mean_tgt"]:.0f} chars</span></div>
      </div>
      <div class="readout">
        <div><span class="k">{tgt} vs {ref} length</span><span class="v">{v["ratio"]:.2f}&times;</span></div>
        <div><span class="k">P(chose longer), controls</span><span class="v">{v["p_longer"]:.2f}</span></div>
        <div><span class="k">95% CI</span><span class="v">{lo:.2f} &ndash; {hi:.2f}</span></div>
        <div><span class="k">Length vs preference (r)</span><span class="v">{v["r"]:+.2f}</span></div>
      </div>
      <p class="caption">{conclusion}</p>
    </div>"""


def _provenance_svg(p_h: float, p_m: float) -> str:
    """Two points on the same prefers-English..prefers-Afrikaans axis: human Afrikaans
    (teal fill) and machine Afrikaans (amber ring). Their closeness is the message, so the
    two labels are split above/below the axis to stay legible even when the points overlap."""
    w, x0, x1, y = 760, 64, 696, 96
    plot = x1 - x0

    def x(v: float) -> float:
        return x0 + v * plot

    ticks = ""
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        ticks += (
            f'<line x1="{x(v):.1f}" y1="{y}" x2="{x(v):.1f}" y2="{y + 6}" '
            f'stroke="#cbd5e1" stroke-width="1"/>'
            f'<text x="{x(v):.1f}" y="{y + 22}" text-anchor="middle" class="svg-tick">{v:.2f}</text>'
        )
    return f"""<svg viewBox="0 0 {w} 150" class="effect-svg" role="img"
      aria-label="Human vs machine Afrikaans, both against English">
  <rect x="{x0}" y="{y - 11}" width="{(x(0.5) - x0):.1f}" height="22" fill="{TEAL}" opacity="0.09"/>
  <rect x="{x(0.5):.1f}" y="{y - 11}" width="{(x1 - x(0.5)):.1f}" height="22" fill="{AMBER}" opacity="0.11"/>
  <line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#94a3b8" stroke-width="1.5"/>
  {ticks}
  <text x="{x0}" y="{y - 50}" text-anchor="start" class="svg-end" fill="{TEAL_DARK}">&#9664; prefers English</text>
  <text x="{x1}" y="{y - 50}" text-anchor="end" class="svg-end" fill="{AMBER_DARK}">prefers Afrikaans &#9654;</text>
  <line x1="{x(0.5):.1f}" y1="{y - 38}" x2="{x(0.5):.1f}" y2="{y + 10}"
        stroke="#64748b" stroke-width="1.5" stroke-dasharray="4 3"/>
  <text x="{x(0.5):.1f}" y="{y - 26}" text-anchor="middle" class="svg-null">cannot tell apart (0.50)</text>
  <circle cx="{x(p_h):.1f}" cy="{y}" r="8" fill="{TEAL}" stroke="white" stroke-width="2"/>
  <text x="{x(p_h):.1f}" y="{y - 16}" text-anchor="middle" class="svg-prov" fill="{TEAL_DARK}">human {p_h:.2f}</text>
  <circle cx="{x(p_m):.1f}" cy="{y}" r="8" fill="white" stroke="{AMBER_DARK}" stroke-width="2.5"/>
  <text x="{x(p_m):.1f}" y="{y + 42}" text-anchor="middle" class="svg-prov" fill="{AMBER_DARK}">machine {p_m:.2f}</text>
</svg>"""


def _provenance_card(human: dict, machine: dict | None, h2h: dict | None) -> str:
    """Human vs machine Afrikaans: is the Afrikaans penalty a translation-quality artefact,
    or the language itself? `human` is afr-vs-en, `machine` is afr_mt-vs-en, `h2h` is
    afr_mt-vs-afr. Renders only when a machine-Afrikaans arm exists in the run."""
    if not machine or not h2h:
        return ""
    if pd.isna(machine.get("p_prefer_target")) or pd.isna(h2h.get("p_prefer_target")):
        return ""

    p_h = human["p_prefer_target"]
    p_m = machine["p_prefer_target"]
    p_hm = h2h["p_prefer_target"]
    lo, hi = h2h["ci95"]
    both_dispreferred = p_h < 0.5 and p_m < 0.5
    indistinct = not h2h["significant"]

    if both_dispreferred and indistinct:
        verdict_lead = "no reliable difference"
        conclusion = (
            "A skeptic could argue the Afrikaans penalty is really a <em>translation-quality</em> "
            "penalty: perhaps the human Afrikaans simply reads a little awkwardly. So we added an "
            "<strong>independent machine translation</strong>. If quality drove the effect, the two "
            "should part ways. They do not: machine Afrikaans is penalised against English just as "
            f"human Afrikaans is ({p_m:.2f} vs {p_h:.2f}), and head-to-head the model shows "
            f"<strong>no reliable preference</strong> between them ({p_hm:.2f}, 95% CI spanning "
            "0.50). The model is responding to the <strong>language</strong>, not the translation."
        )
    elif indistinct:
        verdict_lead = "no reliable difference"
        conclusion = (
            f"Head-to-head, the model shows no reliable preference between human and machine "
            f"Afrikaans ({p_hm:.2f}, 95% CI spanning 0.50), so the two translation sources behave "
            "alike on this run."
        )
    else:
        favoured = "human" if p_hm < 0.5 else "machine"
        verdict_lead = f"{favoured} Afrikaans favoured"
        conclusion = (
            f"Head-to-head the model favours <strong>{favoured}</strong> Afrikaans "
            f"({p_hm:.2f}, 95% CI [{lo:.2f}, {hi:.2f}]), so translation source is itself a factor "
            "here and the language effect cannot be fully separated from translation quality."
        )

    return f"""
    <div class="card">
      <h2>Human or machine translation &mdash; does it matter?</h2>
      <p class="sub">Both a human and an independently machine-translated Afrikaans version of every
        testimony were run against the English source. If the penalty were about translation
        <em>quality</em>, the two would diverge. Here is where each lands against English:</p>
      {_provenance_svg(p_h, p_m)}
      <div class="legend">
        <span><span class="dot fill"></span>Human Afrikaans vs English <b>{p_h:.2f}</b></span>
        <span><span class="dot ring"></span>Machine Afrikaans vs English <b>{p_m:.2f}</b></span>
      </div>
      <div class="h2h">
        <div class="h2h-num">{p_hm:.2f}</div>
        <div class="h2h-cap"><span class="h2h-tag">{verdict_lead}</span>
          P(model prefers <b>machine</b> over <b>human</b> Afrikaans), same content and language.
          0.50 means it cannot tell them apart. 95% CI [{lo:.2f}, {hi:.2f}].</div>
      </div>
      <p class="caption">{conclusion}</p>
    </div>"""


CSS = f"""
:root {{ --ink:#16302b; --muted:#5f7470; --line:#e4ebe8; --card:#ffffff;
  --teal:{TEAL}; --teal-dark:{TEAL_DARK}; --amber:{AMBER}; --amber-dark:{AMBER_DARK}; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:#f3f6f4; color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.55; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:960px; margin:0 auto; padding:0 24px 64px; }}
.hero {{ background:
    radial-gradient(1100px 520px at 18% -8%, rgba(91,168,147,.42) 0%, rgba(91,168,147,0) 60%),
    radial-gradient(900px 600px at 100% 120%, rgba(224,162,74,.20) 0%, rgba(224,162,74,0) 55%),
    linear-gradient(125deg, #12433b 0%, #123b3b 42%, #112b30 100%);
  color:#f4f7f5; padding:48px 0 70px; }}
.hero .wrap {{ padding-bottom:0; }}
.brandbar {{ display:flex; align-items:center; gap:14px; margin-bottom:30px; }}
.brandbar img {{ width:52px; height:52px; }}
.brandbar .name {{ font-weight:800; font-size:1.15rem; letter-spacing:.01em; }}
.brandbar .name span {{ display:block; font-weight:600; font-size:.72rem; letter-spacing:.18em;
  text-transform:uppercase; color:var(--teal); margin-top:2px; }}
.hero h1 {{ font-size:2.5rem; line-height:1.1; margin:0 0 14px; font-weight:800; max-width:760px; }}
.hero p.lead {{ font-size:1.15rem; color:#cfe0da; margin:0; max-width:680px; }}
.hero-stat {{ display:flex; align-items:center; gap:26px; margin-top:38px; flex-wrap:wrap; }}
.hero-stat .big {{ font-size:4.6rem; font-weight:800; line-height:1; letter-spacing:-.02em;
  font-variant-numeric:tabular-nums; color:#fff; }}
.hero-stat .cap {{ font-size:.98rem; color:#cfe0da; max-width:380px; }}
.badge {{ display:inline-block; background:{TEAL_LIGHT}; color:#0c352e; font-weight:700;
  font-size:.8rem; padding:6px 13px; border-radius:999px; margin-top:14px; }}
.chips {{ display:flex; flex-wrap:wrap; gap:12px; margin:-34px 0 8px; position:relative; }}
.chip {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:13px 17px; min-width:120px; box-shadow:0 4px 14px rgba(18,67,59,.07); }}
.chip .n {{ font-size:1.5rem; font-weight:800; font-variant-numeric:tabular-nums; color:var(--teal-dark); }}
.chip .l {{ font-size:.76rem; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:16px;
  padding:28px 30px; margin:22px 0; box-shadow:0 1px 3px rgba(18,67,59,.05); }}
.card h2 {{ font-size:1.3rem; margin:0 0 4px; }}
.card .sub {{ color:var(--muted); margin:0 0 18px; font-size:.95rem; }}
.effect-svg {{ width:100%; height:auto; }}
.svg-tick {{ fill:#94a3b8; font-size:12px; }}
.svg-null {{ fill:#475569; font-size:12px; font-weight:600; }}
.svg-end {{ font-size:12.5px; font-weight:700; }}
.svg-point {{ fill:var(--ink); font-size:16px; font-weight:800; }}
.legend {{ display:flex; gap:22px; flex-wrap:wrap; margin-top:6px; font-size:.84rem; color:#475569; }}
.legend b {{ color:var(--ink); font-variant-numeric:tabular-nums; }}
.dot {{ display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:7px;
  vertical-align:middle; }}
.dot.fill {{ background:var(--teal); }}
.dot.ring {{ background:#fff; border:2.5px solid var(--amber-dark); }}
.dot.bar {{ width:4px; height:14px; border-radius:2px; background:var(--teal-dark); }}
.caption {{ margin:16px 0 0; padding:13px 16px; background:#f0f7f4; border-radius:10px;
  font-size:.92rem; color:#2f5249; }}
.readout {{ display:flex; gap:28px; flex-wrap:wrap; margin-top:18px; padding-top:18px;
  border-top:1px solid var(--line); }}
.readout div span {{ display:block; }}
.readout .k {{ color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.05em; }}
.readout .v {{ font-size:1.25rem; font-weight:700; font-variant-numeric:tabular-nums; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; }}
.gauge .n {{ font-size:2.4rem; font-weight:800; font-variant-numeric:tabular-nums; color:var(--teal-dark); }}
.gauge .l {{ color:var(--muted); font-size:.9rem; }}
.scn-legend {{ display:flex; justify-content:space-between; font-size:.78rem; margin-bottom:14px;
  font-weight:700; }}
.scn-legend .l-en {{ color:var(--teal-dark); }}
.scn-legend .l-afr {{ color:var(--amber-dark); }}
.scn-row {{ display:grid; grid-template-columns:210px 1fr 52px; align-items:center; gap:14px;
  margin:7px 0; }}
.scn-name {{ font-size:.86rem; color:#334155; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
.scn-track {{ position:relative; height:18px; background:#eef3f1; border-radius:5px; }}
.scn-center {{ position:absolute; left:50%; top:-3px; bottom:-3px; width:2px; background:#94a3b8; }}
.scn-bar {{ position:absolute; top:0; bottom:0; border-radius:4px; }}
.scn-bar.neg {{ background:{TEAL}; }}
.scn-bar.pos {{ background:{AMBER}; }}
.scn-val {{ text-align:right; font-variant-numeric:tabular-nums; font-weight:700; font-size:.86rem; }}
.svg-prov {{ font-size:13px; font-weight:800; }}
.lad-lab {{ fill:#334155; font-size:14px; font-weight:700; }}
.lad-prov {{ fill:#94a3b8; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }}
.lad-val {{ fill:var(--ink); font-size:14px; font-weight:800; font-variant-numeric:tabular-nums; }}
.h2h {{ display:flex; align-items:center; gap:22px; margin-top:20px; padding:18px 22px;
  background:#f0f7f4; border-radius:12px; }}
.h2h-num {{ font-size:3rem; font-weight:800; line-height:1; font-variant-numeric:tabular-nums;
  color:var(--teal-dark); }}
.h2h-cap {{ font-size:.95rem; color:#2f5249; }}
.h2h-tag {{ display:inline-block; background:{TEAL_LIGHT}; color:#0c352e; font-weight:700;
  font-size:.74rem; text-transform:uppercase; letter-spacing:.04em; padding:3px 10px;
  border-radius:999px; margin-right:8px; }}
.lenbars {{ margin:6px 0 4px; }}
.lenbar-row {{ display:grid; grid-template-columns:96px 1fr 92px; align-items:center; gap:14px; margin:9px 0; }}
.lenbar-lab {{ font-size:.86rem; font-weight:700; color:#334155; }}
.lenbar {{ height:20px; background:#eef3f1; border-radius:5px; overflow:hidden; }}
.lenbar-fill {{ height:100%; border-radius:5px; }}
.lenbar-fill.en {{ background:var(--teal); }}
.lenbar-fill.afr {{ background:var(--amber); }}
.lenbar-num {{ font-size:.84rem; text-align:right; font-variant-numeric:tabular-nums; color:#475569; }}
.note {{ border-left:4px solid var(--teal); background:#eef7f3; padding:16px 20px;
  border-radius:0 10px 10px 0; margin:22px 0; }}
.note h3 {{ margin:0 0 6px; font-size:1.05rem; }}
.note p {{ margin:0; color:#334155; font-size:.94rem; }}
ul.caveats {{ margin:8px 0 0; padding-left:20px; color:#475569; font-size:.92rem; }}
ul.caveats li {{ margin:6px 0; }}
footer {{ color:var(--muted); font-size:.84rem; margin-top:36px; text-align:center; }}
@media (max-width:680px){{ .grid2{{grid-template-columns:1fr;}} .scn-row{{grid-template-columns:130px 1fr 46px;}}
  .hero h1{{font-size:1.9rem;}} .hero-stat .big{{font-size:3.2rem;}} }}
"""


def build_html(
    summary: dict,
    baselines: dict,
    effect: dict,
    effects_df: pd.DataFrame,
    meta: dict,
    n_scenarios: int,
    epochs: int,
    languages: list[str],
    reports: list[dict],
    verbosity: dict | None = None,
    machine: dict | None = None,
    h2h: dict | None = None,
) -> str:
    p = effect["p_prefer_target"]
    lo, hi = effect["ci95"]
    cont = effect["p_prefer_target_continuous"]
    prefers_en = (1 - p) * 100
    sig = effect["significant"]
    logo = _logo_data_uri()
    model_short = meta["model"].split("/")[-1]
    badge = (
        f'<span class="badge">Statistically significant &mdash; '
        f"95% CI [{lo:.3f}, {hi:.3f}]</span>"
        if sig
        else '<span class="badge" style="background:#fcd34d">Not significant</span>'
    )

    lang_chip = " / ".join(languages) if languages else "en / afr"
    chips = "".join(
        f'<div class="chip"><div class="n">{n}</div><div class="l">{lbl}</div></div>'
        for n, lbl in [
            (f"{summary['n_trials']:,}", "trials"),
            (n_scenarios, "scenarios"),
            (lang_chip, "languages"),
            (epochs, "epochs / trial"),
            (effect["n_cross_trials"], "cross-language"),
            (f"{summary['n_parse_errors']}", "parse errors"),
        ]
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ufakazi &mdash; Language-Conditioned Truthiness Bias</title>
<style>{CSS}</style></head>
<body>
  <div class="hero"><div class="wrap">
    <div class="brandbar">
      <img src="{logo}" alt="Ufakazi logo">
      <div class="name">Ufakazi<span>Language-bias audit</span></div>
    </div>
    <h1>Does an LLM judge testimony by the language it is written in?</h1>
    <p class="lead">Given two contradictory, evidence-balanced witness statements, we hold the
      content and reading order fixed and vary only the <strong>language</strong> each statement
      is written in. A fair model should not care.</p>
    <div class="hero-stat">
      <div class="big">{p * 100:.1f}%</div>
      <div class="cap">of the time, <strong>{model_short}</strong> preferred the
        <strong>Afrikaans</strong>-written testimony over the identical English one.
        50% would mean no language bias &mdash; so it chose <strong>English {prefers_en:.0f}%</strong>
        of the time.<br>{badge}</div>
    </div>
  </div></div>

  <div class="wrap">
    <div class="chips">{chips}</div>

    <div class="card">
      <h2>The language main effect</h2>
      <p class="sub">Cross-language trials only, aggregated symmetrically so that each
        testimony's content appears in each language equally &mdash; this cancels content bias and
        isolates the effect of language alone. The bar is the 95% scenario-bootstrap interval.</p>
      {_main_effect_svg(p, lo, hi, cont)}
      <div class="legend">
        <span><span class="dot fill"></span>Binary choice <b>{p:.3f}</b></span>
        <span><span class="dot ring"></span>Continuous, logprob <b>{cont:.3f}</b></span>
        <span><span class="dot bar"></span>95% bootstrap CI <b>{lo:.3f}&ndash;{hi:.3f}</b></span>
      </div>
      <p class="caption">Two independent measures, one verdict: the binary choice rate
        (<b>{p:.3f}</b>) and the continuous logprob mass on Afrikaans (<b>{cont:.3f}</b>) agree,
        and the whole interval sits well left of the 0.50 no-preference line.</p>
      <div class="readout">
        <div><span class="k">Binary choice</span><span class="v">{p:.3f}</span></div>
        <div><span class="k">Continuous (logprob)</span><span class="v">{cont:.3f}</span></div>
        <div><span class="k">95% CI</span><span class="v">{lo:.3f} &ndash; {hi:.3f}</span></div>
        <div><span class="k">Trials</span><span class="v">{effect["n_cross_trials"]}</span></div>
      </div>
    </div>
{_language_gradient_card(reports, effect["reference"])}
    <div class="grid2">
      <div class="card gauge">
        <h2>Position baseline</h2>
        <p class="sub">Same-language controls.</p>
        <div class="n">{baselines["position_first_rate"]:.2f}</div>
        <div class="l">chose the first-presented testimony (0.50 = no position bias)</div>
      </div>
      <div class="card gauge">
        <h2>Content baseline</h2>
        <p class="sub">Same-language controls.</p>
        <div class="n">{baselines["content_pref_A_rate"]:.2f}</div>
        <div class="l">preferred testimony A on content alone (0.50 = balanced)</div>
      </div>
    </div>

    <div class="card">
      <h2>Where the effect comes from</h2>
      <p class="sub">Per-scenario language shift: how much writing testimony A in Afrikaans changes
        the chance it is chosen, content held fixed. Negative means Afrikaans lowered credibility.</p>
      <div class="scn-legend"><span class="l-en">&#9664; Afrikaans lowered credibility</span>
        <span class="l-afr">Afrikaans raised credibility &#9654;</span></div>
      {_scenario_bars(effects_df)}
    </div>
{_verbosity_card(verbosity, effect["p_prefer_target"])}
{_provenance_card(effect, machine, h2h)}
    <div class="note">
      <h3>Why every trial is run ten times</h3>
      <p>This model is <strong>not deterministic even at temperature 0</strong>: identical prompts
        split roughly 60/40 across the two choices, and pinning the provider and setting a seed did
        not fix it (the noise is upstream, in batched floating-point and routing). So we replicate
        each trial across <strong>{epochs} epochs</strong> and report bootstrap confidence intervals
        rather than a single brittle number.</p>
    </div>

    <div class="card">
      <h2>Read with care</h2>
      <ul class="caveats">
        <li>Scenarios and testimonies are <strong>synthetic</strong>; only aggregate rates are shown.</li>
        <li>One model so far (<strong>{meta["model"]}</strong>); the effect may not generalise.</li>
        <li>For <strong>Afrikaans</strong>, human and machine translations behave alike (see above),
          separating the language effect from translation quality. The <strong>isiZulu and isiXhosa</strong>
          arms are machine-translated and unverified, so their steeper penalty conflates language with
          possible translation degradation; a human-verified pass is the next step there.</li>
        <li>The aggregate effect is bias-cancelled; the <em>per-scenario</em> bars are indicative
          only, since a single scenario's content imbalance is not cancelled in isolation.</li>
      </ul>
    </div>

    <footer>Generated from <code>{meta["model"]}</code> &middot; run {meta["created"]} &middot;
      {summary["n_trials"]:,} trials &middot; reproduce with
      <code>uv run python artefacts/generate_report.py</code></footer>
  </div>
</body></html>"""


def main() -> None:
    df = filter_latest_eval(load_trials(LOG_DIR))
    if df.empty:
        raise SystemExit("No trials found in results/logs. Run `ufakazi run` first.")
    summary = summarize(df)
    baselines = control_baselines(df)
    reports = language_report(df)
    if not reports:
        raise SystemExit("No cross-language trials in the latest run.")
    # Headline effect is human Afrikaans vs English; afr_mt (machine) is a robustness arm.
    effect = next((r for r in reports if r["target"] == "afr"), reports[0])
    target, reference = effect["target"], effect["reference"]
    effects_df = per_scenario_language_effect(df, target, reference)
    meta = _latest_eval_meta(df)
    n_scenarios = int(df["metadata_scenario_id"].nunique())
    epochs = int(df["epoch"].nunique()) if "epoch" in df.columns else 1
    languages = [reference] + [r["target"] for r in reports]

    verbosity = _verbosity_facts(df, target, reference)

    # Machine-Afrikaans robustness arm: afr_mt vs en, plus the direct human-vs-machine contrast.
    machine = next((r for r in reports if r["target"] == "afr_mt"), None)
    h2h = language_preference(df, "afr_mt", "afr") if machine else None

    html = build_html(
        summary,
        baselines,
        effect,
        effects_df,
        meta,
        n_scenarios,
        epochs,
        languages,
        reports,
        verbosity,
        machine,
        h2h,
    )
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT}  ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
