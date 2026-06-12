"""Hand-rolled rendering helpers — no extra dependencies.

Three small renderers the GUI needs and we refuse to pull a library for:

- :func:`markdown_to_html` — the fixed subset our own wiki generator emits
  (headings, lists, tables, links, bold/italic/code, paragraphs). It is *not*
  a general Markdown engine; it only has to read what :mod:`memory.wiki` writes.
- :func:`emit_toml` — serialize a :class:`Config` back to the flat schema the
  loader accepts. Comments are not preserved (regenerated from current values).
- :func:`sparkline_svg` — a hand-drawn polyline of a goal's checkpoint arc.

Everything here escapes untrusted text; the wiki is generated, but findings
carry user/model prose, so HTML escaping is non-negotiable.
"""

from __future__ import annotations

import html
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dexta_intelligence.config import Config

__all__ = ["emit_toml", "markdown_to_html", "sparkline_svg"]


# ── markdown (our fixed subset) ───────────────────────────────────────────────

_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*([^*]+)\*(?![*\w])")
_ITALIC_US = re.compile(r"(?<![_\w])_([^_]+)_(?![_\w])")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

#: Only these link schemes are allowed; anything else (javascript:, data:, …)
#: is neutralised so model-authored prose can't emit a live dangerous link.
_SAFE_SCHEMES = ("http://", "https://", "mailto:")


def _safe_href(url: str) -> str:
    """Allow http/https/mailto and relative/anchor links; neutralise the rest."""
    lowered = url.strip().lower()
    if lowered.startswith(_SAFE_SCHEMES):
        return url
    # Relative paths and same-page anchors carry no scheme — keep them.
    if ":" not in lowered.split("/", 1)[0] and not lowered.startswith("//"):
        return url
    return "#"


def _inline(text: str) -> str:
    """Escape, then re-introduce the small set of inline spans we support."""
    out = html.escape(text)
    # Code first — its contents are literal (no further inline parsing).
    out = _INLINE_CODE.sub(lambda m: f"<code>{m.group(1)}</code>", out)
    out = _LINK.sub(lambda m: f'<a href="{_safe_href(m.group(2))}">{m.group(1)}</a>', out)
    out = _BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", out)
    out = _ITALIC.sub(lambda m: f"<em>{m.group(1)}</em>", out)
    out = _ITALIC_US.sub(lambda m: f"<em>{m.group(1)}</em>", out)
    return out


def _table_row(line: str, *, header: bool) -> str:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    tag = "th" if header else "td"
    body = "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells)
    return f"<tr>{body}</tr>"


def _is_table_divider(line: str) -> bool:
    stripped = line.strip().strip("|")
    if not stripped:
        return False
    cells = [c.strip() for c in stripped.split("|")]
    return all(cell and set(cell) <= {"-", ":"} for cell in cells)


def markdown_to_html(md: str) -> str:  # noqa: PLR0915 - a single sequential line scanner
    """Render the fixed Markdown subset our wiki emits into safe HTML.

    Supports: ATX headings, unordered lists (``-``) with one nesting level,
    pipe tables, links, bold/italic/inline-code, and paragraphs. Anything else
    is treated as paragraph text (and escaped).
    """
    lines = md.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i = 0
    list_stack: list[int] = []  # indent widths of open <ul>s

    def close_lists(to_depth: int = 0) -> None:
        while len(list_stack) > to_depth:
            out.append("</ul>")
            list_stack.pop()

    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            close_lists()
            i += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            close_lists()
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            i += 1
            continue

        # Table: a header row followed by a divider row.
        if "|" in line and i + 1 < n and _is_table_divider(lines[i + 1]):
            close_lists()
            out.append("<table>")
            out.append(f"<thead>{_table_row(line, header=True)}</thead>")
            out.append("<tbody>")
            i += 2
            while i < n and "|" in lines[i] and lines[i].strip():
                out.append(_table_row(lines[i], header=False))
                i += 1
            out.append("</tbody></table>")
            continue

        bullet = re.match(r"^(\s*)-\s+(.*)$", line)
        if bullet:
            indent = len(bullet.group(1))
            if not list_stack or indent > list_stack[-1]:
                out.append("<ul>")
                list_stack.append(indent)
            else:
                while len(list_stack) > 1 and indent < list_stack[-1]:
                    out.append("</ul>")
                    list_stack.pop()
            out.append(f"<li>{_inline(bullet.group(2))}</li>")
            i += 1
            continue

        # Plain paragraph.
        close_lists()
        out.append(f"<p>{_inline(stripped)}</p>")
        i += 1

    close_lists()
    return "\n".join(out)


# ── TOML emitter (flat known schema) ──────────────────────────────────────────


def _toml_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def emit_toml(config: Config) -> str:
    """Serialize a Config to the flat TOML schema ``load_config`` accepts.

    Comments are regenerated, not preserved — the panel writes the *current*
    state. Secrets are intentionally written as the empty strings they hold in
    a shared-safe config (the GUI never persists env-sourced secrets to disk).
    """
    a = config.analysis
    w = config.wiki
    llm = config.llm
    d = config.data
    lines = [
        "# dexta-intelligence configuration",
        "# Regenerated by `dexta serve` settings. Secrets belong in the environment.",
        "",
        "[data]",
        f"backend = {_toml_str(d.backend)}",
        f"sqlite_path = {_toml_str(str(d.sqlite_path))}",
        "",
        "[llm]",
        f"provider = {_toml_str(llm.provider)}",
        f"model = {_toml_str(llm.model)}",
        "",
        "[analysis]",
        f"target_low = {a.target_low}",
        f"target_high = {a.target_high}",
        f"deep_analysis_window_days = {a.deep_analysis_window_days}",
        "",
        "[wiki]",
        f"path = {_toml_str(str(w.path))}",
        f"git = {'true' if w.git else 'false'}",
        "",
    ]
    return "\n".join(lines)


# ── SVG sparkline (goal checkpoint arc) ───────────────────────────────────────


def sparkline_svg(values: Sequence[float], *, width: int = 240, height: int = 48) -> str:
    """A hand-rolled polyline of a goal's metric arc. Empty data ⇒ a flat hint."""
    pad = 4
    pts = [v for v in values if v is not None]
    if len(pts) < 2:
        mid = height / 2
        return (
            f'<svg class="spark" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img" aria-label="not enough checkpoints">'
            f'<line x1="{pad}" y1="{mid:.1f}" x2="{width - pad}" y2="{mid:.1f}" '
            f'class="spark-flat" /></svg>'
        )
    lo, hi = min(pts), max(pts)
    span = (hi - lo) or 1.0
    n = len(pts)
    coords: list[str] = []
    for idx, v in enumerate(pts):
        x = pad + (width - 2 * pad) * idx / (n - 1)
        y = height - pad - (height - 2 * pad) * (v - lo) / span
        coords.append(f"{x:.1f},{y:.1f}")
    last_x, last_y = coords[-1].split(",")
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="metric arc">'
        f'<polyline class="spark-line" points="{" ".join(coords)}" '
        f'fill="none" />'
        f'<circle class="spark-dot" cx="{last_x}" cy="{last_y}" r="3" /></svg>'
    )
