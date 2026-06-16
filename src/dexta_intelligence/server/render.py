"""Hand-rolled rendering helpers — no extra dependencies.

Three small renderers the GUI needs and we refuse to pull a library for:

- :func:`markdown_to_html` — a small, dependency-free general Markdown renderer.
  It covers what our wiki generator emits *and* the prose LLM answers produce
  (headings, ordered/unordered lists, tables, links, bold/italic/code, fenced
  code, blockquotes, horizontal rules, paragraphs). It is escape-first.
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
_BOLD_US = re.compile(r"(?<!\w)__([^_]+)__(?!\w)")
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
    out = _BOLD_US.sub(lambda m: f"<strong>{m.group(1)}</strong>", out)
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


_HR = re.compile(r"^(-{3,}|\*{3,}|_{3,})$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_BULLET = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_ORDERED = re.compile(r"^(\s*)\d+[.)]\s+(.*)$")
_BLOCKQUOTE = re.compile(r"^\s*>\s?(.*)$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def markdown_to_html(md: str) -> str:  # noqa: PLR0912, PLR0915 - one sequential scanner
    """Render Markdown into safe HTML with no third-party dependencies.

    Supports ATX headings, ordered and unordered lists (one nesting level),
    fenced code blocks, blockquotes, horizontal rules, pipe tables, links,
    bold/italic/inline-code, and paragraphs. Everything is escaped first; only
    the recognised constructs re-introduce markup.
    """
    lines = md.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    # Each open list is (indent, tag) where tag is "ul" or "ol".
    list_stack: list[tuple[int, str]] = []

    def close_lists(to_depth: int = 0) -> None:
        while len(list_stack) > to_depth:
            out.append(f"</{list_stack.pop()[1]}>")

    def flush_para(buf: list[str]) -> None:
        if buf:
            out.append(f"<p>{'<br>'.join(_inline(s) for s in buf)}</p>")
            buf.clear()

    para: list[str] = []
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_para(para)
            close_lists()
            i += 1
            continue

        fence = _FENCE.match(line)
        if fence:
            flush_para(para)
            close_lists()
            marker = fence.group(1)
            i += 1
            code: list[str] = []
            while i < n and not lines[i].strip().startswith(marker):
                code.append(lines[i])
                i += 1
            i += 1  # consume the closing fence (or run off the end)
            out.append(f"<pre><code>{html.escape(chr(10).join(code))}</code></pre>")
            continue

        if _HR.match(stripped):
            flush_para(para)
            close_lists()
            out.append("<hr>")
            i += 1
            continue

        heading = _HEADING.match(stripped)
        if heading:
            flush_para(para)
            close_lists()
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            i += 1
            continue

        if "|" in line and i + 1 < n and _is_table_divider(lines[i + 1]):
            flush_para(para)
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

        quote = _BLOCKQUOTE.match(line)
        if quote:
            flush_para(para)
            close_lists()
            block: list[str] = []
            while i < n and (m := _BLOCKQUOTE.match(lines[i])):
                block.append(m.group(1))
                i += 1
            inner = markdown_to_html("\n".join(block))
            out.append(f"<blockquote>{inner}</blockquote>")
            continue

        bullet = _BULLET.match(line)
        ordered = _ORDERED.match(line) if not bullet else None
        match = bullet or ordered
        if match is not None:
            flush_para(para)
            tag = "ul" if bullet else "ol"
            indent = len(match.group(1))
            if not list_stack or indent > list_stack[-1][0]:
                out.append(f"<{tag}>")
                list_stack.append((indent, tag))
            else:
                while len(list_stack) > 1 and indent < list_stack[-1][0]:
                    out.append(f"</{list_stack.pop()[1]}>")
            out.append(f"<li>{_inline(match.group(2))}</li>")
            i += 1
            continue

        close_lists()
        para.append(stripped)
        i += 1

    flush_para(para)
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
