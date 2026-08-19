"""Assemble the blog page (docs/index.html): figures inlined, Sources list generated.

Citations in the post are **linked claims**, not markers: the phrase that makes the
claim is an `<a class="src" href=... data-cite="...">`. `data-cite` is the human-readable
reference (who, title, publisher, date). This module walks the page in document order,
collects every `.src` link, dedupes by URL (first appearance wins), and writes them as a
numbered list between `<!-- sources -->` and `<!-- /sources -->`, so nothing cited in the
body can be missing from the list and nothing in the list can be uncited. A `.src` link
without `data-cite` fails the build rather than silently producing a bare URL.

    uv run python -m ufakazi.site            # inline figures + regenerate Sources
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from urllib.parse import urlparse

PAGE = Path("docs/index.html")
FIG_DIR = Path("docs/figures")
SOURCES_START = "<!-- sources -->"
SOURCES_END = "<!-- /sources -->"

# Matches an anchor carrying class "src" (among possibly other classes), capturing its
# attribute string; attributes are then read individually so their order does not matter.
_SRC_ANCHOR = re.compile(r"<a\b([^>]*\bclass=\"[^\"]*\bsrc\b[^\"]*\"[^>]*)>", re.S)
_ATTR = re.compile(r"\b([\w-]+)=\"([^\"]*)\"")


def collect_sources(page_html: str) -> list[dict]:
    """Every distinct `.src` link in document order: {href, cite, domain}."""
    seen: dict[str, dict] = {}
    for m in _SRC_ANCHOR.finditer(page_html):
        attrs = dict(_ATTR.findall(m.group(1)))
        href = attrs.get("href")
        cite = attrs.get("data-cite")
        if not href:
            raise ValueError(f"src link without href: {m.group(0)[:120]}")
        if not cite:
            raise ValueError(f"src link without data-cite: {href}")
        if href in seen:
            continue
        seen[href] = {
            "href": href,
            "cite": html.unescape(cite),
            "domain": urlparse(href).netloc.removeprefix("www."),
        }
    return list(seen.values())


def render_sources(page_html: str) -> tuple[str, int]:
    """Replace the block between the source markers with the generated list."""
    start, end = page_html.find(SOURCES_START), page_html.find(SOURCES_END)
    if start < 0 or end < 0:
        raise ValueError("page has no <!-- sources --> ... <!-- /sources --> block")
    sources = collect_sources(page_html)
    items = "\n".join(
        f'    <li id="src-{i}">{html.escape(s["cite"], quote=False)}. '
        f'<a href="{s["href"]}">{html.escape(s["domain"])}</a></li>'
        for i, s in enumerate(sources, 1)
    )
    block = (
        f'{SOURCES_START}\n  <ol class="sources">\n{items}\n  </ol>\n  {SOURCES_END}'
    )
    return page_html[:start] + block + page_html[end + len(SOURCES_END) :], len(sources)


def build(page: Path = PAGE, fig_dir: Path = FIG_DIR) -> None:
    from ufakazi.figures_web import inline_figures

    done = inline_figures(page, fig_dir)
    print(f"figures: inlined {len(done)}: {', '.join(done)}")
    text, n = render_sources(page.read_text())
    page.write_text(text)
    print(f"sources: {n} listed")


if __name__ == "__main__":
    build()
