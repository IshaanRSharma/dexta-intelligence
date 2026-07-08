// Temporal episode graph renderer for the Timeline page.
//
// Vanilla JS, inline SVG, no external libraries. It fetches the deterministic
// episode graph from /episodes.json and draws it: highs above a centre baseline,
// lows below, sensor gaps as hatched bands, and typed context edges (meals,
// boluses, activity, sleep) as glyphs on a lane with thin connectors to the
// episodes they belong to. Nothing here is authored by a model; it only lays out
// the facts the server computed.

(function () {
  "use strict";

  const chartEl = document.getElementById("tl-chart");
  const overviewEl = document.getElementById("tl-overview");
  const tooltipEl = document.getElementById("tl-tooltip");
  const detailEl = document.getElementById("tl-detail");
  const shellEl = document.querySelector(".tl-shell");
  const resetBtn = document.getElementById("tl-reset");
  if (!chartEl || !overviewEl || !shellEl) return;

  const SVGNS = "http://www.w3.org/2000/svg";
  const W = 960;
  const PAD_L = 52;
  const PAD_R = 16;
  const PLOT_W = W - PAD_L - PAD_R;
  const LANE_Y = 30;
  const REGION_TOP = 52;
  const REGION_HALF = 150;
  const CENTER_Y = REGION_TOP + REGION_HALF;
  const AXIS_Y = CENTER_Y + REGION_HALF + 8;
  const H = AXIS_Y + 26;
  const OV_H = 46;
  const MIN_SPAN_MS = 30 * 60 * 1000;

  const CONTEXT_KINDS = ["meal", "bolus", "activity", "sleep"];
  const KIND_TITLE = { hypo: "Low episode", hyper: "High episode", sensor_gap: "Sensor gap" };
  const EXTREME_NOUN = { hypo: "nadir", hyper: "peak" };

  const state = {
    data: null,
    fullStart: 0,
    fullEnd: 0,
    viewStart: 0,
    viewEnd: 0,
    selectedId: null,
    filters: {
      hypo: true, hyper: true, sensor_gap: true,
      meal: true, bolus: true, activity: true, sleep: true, edges: true,
    },
  };

  function svg(tag, attrs) {
    const node = document.createElementNS(SVGNS, tag);
    if (attrs) {
      for (const k in attrs) {
        if (attrs[k] != null) node.setAttribute(k, String(attrs[k]));
      }
    }
    return node;
  }

  function ms(iso) {
    const t = Date.parse(iso);
    return Number.isNaN(t) ? null : t;
  }

  function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  function pad2(n) {
    return n < 10 ? "0" + n : String(n);
  }

  function fmtDay(t) {
    const d = new Date(t);
    return MONTHS[d.getUTCMonth()] + " " + d.getUTCDate();
  }

  function fmtTime(t) {
    const d = new Date(t);
    return pad2(d.getUTCHours()) + ":" + pad2(d.getUTCMinutes());
  }

  function fmtStamp(t) {
    return fmtDay(t) + " " + fmtTime(t);
  }

  function num(v) {
    return typeof v === "number" && isFinite(v);
  }

  function trimNum(v) {
    return num(v) ? String(Math.round(v * 100) / 100) : "";
  }

  // ── scales ────────────────────────────────────────────────────────────────
  function xView(t) {
    const span = state.viewEnd - state.viewStart || 1;
    return PAD_L + ((t - state.viewStart) / span) * PLOT_W;
  }

  function xFull(t) {
    const span = state.fullEnd - state.fullStart || 1;
    return PAD_L + ((t - state.fullStart) / span) * PLOT_W;
  }

  function inView(a, b) {
    return b >= state.viewStart && a <= state.viewEnd;
  }

  function bandHalfHeight(ep) {
    if (ep.severe) return REGION_HALF * 0.92;
    if (ep.clinically_significant) return REGION_HALF * 0.6;
    return REGION_HALF * 0.36;
  }

  // ── episode / edge geometry helpers ─────────────────────────────────────────
  function episodeAnchor(ep) {
    const a = ms(ep.start);
    const b = ms(ep.end);
    const x = clamp((xView(a) + xView(b)) / 2, PAD_L, W - PAD_R);
    if (ep.kind === "hyper") return { x: x, y: CENTER_Y - bandHalfHeight(ep) };
    return { x: x, y: CENTER_Y };
  }

  function edgePath(anchor, gx, gy) {
    const cx = (anchor.x + gx) / 2;
    const cy = (anchor.y + gy) / 2 - 18;
    return "M" + anchor.x + " " + anchor.y + " Q" + cx + " " + cy + " " + gx + " " + gy;
  }

  // ── tooltip ─────────────────────────────────────────────────────────────────
  function showTooltip(html, evt) {
    tooltipEl.innerHTML = html;
    tooltipEl.hidden = false;
    positionTooltip(evt);
  }

  function positionTooltip(evt) {
    if (tooltipEl.hidden) return;
    const shellRect = shellEl.getBoundingClientRect();
    let x = evt.clientX - shellRect.left + 14;
    let y = evt.clientY - shellRect.top + 14;
    const w = tooltipEl.offsetWidth;
    if (x + w > shellRect.width) x = shellRect.width - w - 4;
    tooltipEl.style.left = Math.max(0, x) + "px";
    tooltipEl.style.top = Math.max(0, y) + "px";
  }

  function hideTooltip() {
    tooltipEl.hidden = true;
  }

  function tag(ep) {
    if (ep.severe) return '<span class="tl-tt-tag tl-tt-tag-severe">severe</span>';
    if (ep.clinically_significant) return '<span class="tl-tt-tag tl-tt-tag-sig">clinically significant</span>';
    return "";
  }

  function episodeTooltip(ep) {
    const rows = [];
    rows.push('<div class="tl-tt-title">' + (KIND_TITLE[ep.kind] || "Episode") + "</div>");
    rows.push('<div class="tl-tt-row">' + fmtStamp(ms(ep.start)) + " to " + fmtTime(ms(ep.end)) + " UTC</div>");
    if (num(ep.duration_min)) rows.push('<div class="tl-tt-row">' + trimNum(ep.duration_min) + " min</div>");
    if (num(ep.extreme_mg_dl)) {
      rows.push('<div class="tl-tt-row">' + trimNum(ep.extreme_mg_dl) + " mg/dL " + (EXTREME_NOUN[ep.kind] || "extreme") + "</div>");
    }
    const links = ep.links || [];
    if (ep.kind !== "sensor_gap") {
      rows.push('<div class="tl-tt-row">' + links.length + (links.length === 1 ? " context event" : " context events") + "</div>");
    }
    const t = tag(ep);
    if (t) rows.push(t);
    return rows.join("");
  }

  function edgeDetail(kind, detail) {
    detail = detail || {};
    const parts = [];
    if (kind === "meal") {
      if (num(detail.carbs_g)) parts.push(trimNum(detail.carbs_g) + " g carbs");
      if (detail.note) parts.push(String(detail.note));
    } else if (kind === "bolus") {
      if (num(detail.units)) parts.push(trimNum(detail.units) + " U");
      if (detail.automatic) parts.push("automatic");
    } else if (kind === "activity") {
      if (detail.kind) parts.push(String(detail.kind));
      if (num(detail.intensity)) parts.push("intensity " + trimNum(detail.intensity));
    } else if (kind === "sleep") {
      if (num(detail.score)) parts.push("score " + trimNum(detail.score));
    }
    return parts.join(", ");
  }

  function linkTooltip(link) {
    const rows = [];
    rows.push('<div class="tl-tt-title">' + link.kind + "</div>");
    const d = edgeDetail(link.kind, link.detail);
    if (d) rows.push('<div class="tl-tt-row">' + d + "</div>");
    rows.push('<div class="tl-tt-row">' + fmtTime(ms(link.ts)) + " UTC · " + (num(link.offset_min) ? (link.offset_min > 0 ? "+" : "") + trimNum(link.offset_min) + " min" : "") + "</div>");
    return rows.join("");
  }

  // ── detail panel (mirrors _episode_card.html) ───────────────────────────────
  function chip(k, v) {
    const wrap = document.createElement("span");
    wrap.className = "chip-kv";
    const ks = document.createElement("span");
    ks.className = "chip-k";
    ks.textContent = k;
    const vs = document.createElement("span");
    vs.className = "chip-v";
    vs.textContent = v;
    wrap.appendChild(ks);
    wrap.appendChild(vs);
    return wrap;
  }

  function renderDetail(ep) {
    detailEl.innerHTML = "";
    const card = document.createElement("article");
    card.className = "card finding-card episode-card";

    const head = document.createElement("div");
    head.className = "card-head";
    const h2 = document.createElement("h2");
    h2.textContent = KIND_TITLE[ep.kind] || "Episode";
    head.appendChild(h2);
    if (ep.severe) {
      const b = document.createElement("span");
      b.className = "badge bad";
      b.textContent = "severe";
      head.appendChild(b);
    } else if (ep.clinically_significant) {
      const b = document.createElement("span");
      b.className = "badge neutral";
      b.textContent = "clinically significant";
      head.appendChild(b);
    }
    card.appendChild(head);

    const chips = document.createElement("div");
    chips.className = "chips";
    chips.appendChild(chip("when", fmtStamp(ms(ep.start)) + " to " + fmtTime(ms(ep.end))));
    if (num(ep.duration_min)) chips.appendChild(chip("duration", trimNum(ep.duration_min) + " min"));
    if (num(ep.extreme_mg_dl)) {
      chips.appendChild(chip("extreme", trimNum(ep.extreme_mg_dl) + " mg/dL " + (EXTREME_NOUN[ep.kind] || "extreme")));
    }
    card.appendChild(chips);

    const links = ep.links || [];
    if (links.length) {
      const p = document.createElement("p");
      p.className = "muted small";
      p.textContent = "Context around this episode (offset from its start):";
      card.appendChild(p);
      const ec = document.createElement("div");
      ec.className = "chips";
      links.forEach(function (link) {
        const off = num(link.offset_min) ? (link.offset_min > 0 ? "+" : "") + trimNum(link.offset_min) + " min" : "";
        const label = link.kind + (off ? " " + off : "");
        ec.appendChild(chip(label, edgeDetail(link.kind, link.detail) || fmtTime(ms(link.ts))));
      });
      card.appendChild(ec);
    } else if (ep.kind !== "sensor_gap") {
      const p = document.createElement("p");
      p.className = "muted small";
      p.textContent = "No meals, boluses, activity, or sleep were logged around this episode.";
      card.appendChild(p);
    }
    detailEl.appendChild(card);
  }

  function clearDetail() {
    detailEl.innerHTML = '<p class="muted small">Select an episode to see the context linked around it.</p>';
  }

  // ── highlight / selection ────────────────────────────────────────────────────
  function setEpisodeStrong(epId, on) {
    const edges = chartEl.querySelectorAll('[data-edge-ep="' + cssEscape(epId) + '"]');
    edges.forEach(function (e) { e.classList.toggle("tl-edge-on", on); });
  }

  function cssEscape(s) {
    return String(s).replace(/["\\]/g, "\\$&");
  }

  function applySelection() {
    const sel = state.selectedId;
    const bands = chartEl.querySelectorAll(".tl-band, .tl-gap");
    bands.forEach(function (b) {
      const isSel = b.dataset.ep === sel;
      b.classList.toggle("tl-active", isSel);
      b.classList.toggle("tl-dim", sel != null && !isSel);
    });
    const nodes = chartEl.querySelectorAll(".tl-node");
    nodes.forEach(function (n) {
      const related = sel != null && n.dataset.edgeEp === sel;
      n.classList.toggle("tl-dim", sel != null && !related);
    });
    chartEl.querySelectorAll(".tl-edge").forEach(function (e) {
      e.classList.toggle("tl-edge-on", sel != null && e.dataset.edgeEp === sel);
    });
    if (sel != null) {
      const ep = state.data.nodes.find(function (n) { return n.id === sel; });
      if (ep) renderDetail(ep);
    } else {
      clearDetail();
    }
  }

  function selectEpisode(epId) {
    state.selectedId = state.selectedId === epId ? null : epId;
    applySelection();
  }

  // ── main render ──────────────────────────────────────────────────────────────
  function render() {
    chartEl.innerHTML = "";
    const root = svg("svg", { viewBox: "0 0 " + W + " " + H, class: "tl-svg", "aria-hidden": "true" });

    // hatch pattern for sensor gaps
    const defs = svg("defs");
    const pat = svg("pattern", { id: "tl-hatch", width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
    pat.appendChild(svg("line", { x1: 0, y1: 0, x2: 0, y2: 6, class: "tl-hatch-stroke" }));
    defs.appendChild(pat);
    root.appendChild(defs);

    drawAxes(root);
    drawZoneLabels(root);

    // sensor gaps behind bands
    if (state.filters.sensor_gap) {
      state.data.nodes.forEach(function (ep) {
        if (ep.kind === "sensor_gap") drawGap(root, ep);
      });
    }

    // edges (drawn under glyphs and bands so marks stay legible)
    if (state.filters.edges) {
      state.data.nodes.forEach(function (ep) {
        if (ep.kind === "sensor_gap") return;
        if (!state.filters[ep.kind]) return;
        (ep.links || []).forEach(function (link) {
          if (!state.filters[link.kind]) return;
          drawEdge(root, ep, link);
        });
      });
    }

    // context glyphs
    state.data.nodes.forEach(function (ep) {
      if (ep.kind === "sensor_gap") return;
      if (!state.filters[ep.kind]) return;
      (ep.links || []).forEach(function (link) {
        if (!state.filters[link.kind]) return;
        drawGlyph(root, ep, link);
      });
    });

    // excursion bands on top
    state.data.nodes.forEach(function (ep) {
      if (ep.kind === "hypo" && state.filters.hypo) drawBand(root, ep);
      if (ep.kind === "hyper" && state.filters.hyper) drawBand(root, ep);
    });

    chartEl.appendChild(root);
    applySelection();
    renderOverview();
  }

  function drawAxes(root) {
    root.appendChild(svg("line", { x1: PAD_L, y1: CENTER_Y, x2: W - PAD_R, y2: CENTER_Y, class: "tl-baseline" }));
    root.appendChild(svg("line", { x1: PAD_L, y1: AXIS_Y, x2: W - PAD_R, y2: AXIS_Y, class: "tl-axis-line" }));
    const ticks = axisTicks();
    ticks.forEach(function (t) {
      const x = xView(t);
      root.appendChild(svg("line", { x1: x, y1: REGION_TOP, x2: x, y2: AXIS_Y, class: "tl-axis-line", opacity: 0.4 }));
      const label = svg("text", { x: x, y: AXIS_Y + 14, class: "tl-axis-label", "text-anchor": "middle" });
      label.textContent = spansDays() ? fmtDay(t) : fmtTime(t);
      root.appendChild(label);
    });
  }

  function spansDays() {
    return state.viewEnd - state.viewStart > 2 * 24 * 3600 * 1000;
  }

  function axisTicks() {
    const n = 6;
    const out = [];
    for (let i = 0; i <= n; i++) out.push(state.viewStart + ((state.viewEnd - state.viewStart) * i) / n);
    return out;
  }

  function drawZoneLabels(root) {
    const hi = svg("text", { x: PAD_L - 6, y: REGION_TOP + 12, class: "tl-zone-label", "text-anchor": "end" });
    hi.textContent = "High";
    root.appendChild(hi);
    const lo = svg("text", { x: PAD_L - 6, y: CENTER_Y + REGION_HALF - 2, class: "tl-zone-label", "text-anchor": "end" });
    lo.textContent = "Low";
    root.appendChild(lo);
    const lane = svg("text", { x: PAD_L - 6, y: LANE_Y + 4, class: "tl-lane-label", "text-anchor": "end" });
    lane.textContent = "Context";
    root.appendChild(lane);
  }

  function drawGap(root, ep) {
    const a = ms(ep.start);
    const b = ms(ep.end);
    if (!inView(a, b)) return;
    const x0 = clamp(xView(a), PAD_L, W - PAD_R);
    const x1 = clamp(xView(b), PAD_L, W - PAD_R);
    const g = svg("g", { class: "tl-gap", tabindex: 0, role: "button", "data-ep": ep.id });
    g.setAttribute("aria-label", "Sensor gap, " + trimNum(ep.duration_min) + " minutes from " + fmtStamp(a) + " UTC");
    g.appendChild(svg("rect", { x: x0, y: REGION_TOP, width: Math.max(2, x1 - x0), height: 2 * REGION_HALF }));
    if (x1 - x0 > 26) {
      const label = svg("text", { x: (x0 + x1) / 2, y: REGION_TOP + 12, class: "tl-gap-label", "text-anchor": "middle" });
      label.textContent = "gap";
      g.appendChild(label);
    }
    wireEpisode(g, ep);
    root.appendChild(g);
  }

  function drawBand(root, ep) {
    const a = ms(ep.start);
    const b = ms(ep.end);
    if (!inView(a, b)) return;
    const x0 = clamp(xView(a), PAD_L, W - PAD_R);
    const x1 = clamp(xView(b), PAD_L, W - PAD_R);
    const width = Math.max(4, x1 - x0);
    const h = bandHalfHeight(ep);
    const isHyper = ep.kind === "hyper";
    const y = isHyper ? CENTER_Y - h : CENTER_Y;
    const g = svg("g", {
      class: "tl-band tl-band-" + ep.kind + (ep.severe ? " tl-severe" : ""),
      tabindex: 0, role: "button", "data-ep": ep.id,
    });
    g.setAttribute("aria-label", ariaBand(ep));
    g.appendChild(svg("rect", { x: x0, y: y, width: width, height: h, rx: 3, class: "tl-fill" }));

    if (width > 22) {
      const labelY = isHyper ? y - 4 : y + h + 11;
      const label = svg("text", { x: x0 + width / 2, y: labelY, class: "tl-band-label", "text-anchor": "middle" });
      label.textContent = num(ep.extreme_mg_dl) ? trimNum(ep.extreme_mg_dl) : trimNum(ep.duration_min) + "m";
      g.appendChild(label);
      if (ep.severe) {
        const mark = svg("text", { x: x0 + width / 2, y: isHyper ? y - 15 : y + h + 22, class: "tl-severe-mark", "text-anchor": "middle" });
        mark.textContent = "!";
        g.appendChild(mark);
      }
    }
    wireEpisode(g, ep);
    root.appendChild(g);
  }

  function ariaBand(ep) {
    const kind = ep.kind === "hyper" ? "High" : "Low";
    const sev = ep.severe ? "severe " : ep.clinically_significant ? "clinically significant " : "";
    return sev + kind.toLowerCase() + " episode, " + trimNum(ep.extreme_mg_dl) + " mg/dL, "
      + trimNum(ep.duration_min) + " minutes from " + fmtStamp(ms(ep.start)) + " UTC, "
      + (ep.links || []).length + " context events. Press Enter for detail.";
  }

  function drawEdge(root, ep, link) {
    const t = ms(link.ts);
    if (t < state.viewStart || t > state.viewEnd) return;
    const anchor = episodeAnchor(ep);
    const gx = xView(t);
    const path = svg("path", { d: edgePath(anchor, gx, LANE_Y), class: "tl-edge" });
    path.dataset.edgeEp = ep.id;
    root.appendChild(path);
  }

  function drawGlyph(root, ep, link) {
    const t = ms(link.ts);
    if (t < state.viewStart || t > state.viewEnd) return;
    const x = xView(t);
    const g = svg("g", { class: "tl-node tl-node-" + link.kind, tabindex: 0, role: "button" });
    g.dataset.edgeEp = ep.id;
    g.setAttribute("aria-label", link.kind + " at " + fmtTime(t) + " UTC, " + (edgeDetail(link.kind, link.detail) || "no detail"));
    g.appendChild(glyphShape(link.kind, x, LANE_Y));
    wireGlyph(g, ep, link);
    root.appendChild(g);
  }

  function glyphShape(kind, x, y) {
    const r = 5;
    if (kind === "meal") return svg("circle", { cx: x, cy: y, r: r });
    if (kind === "sleep") return svg("rect", { x: x - r, y: y - r, width: 2 * r, height: 2 * r });
    if (kind === "bolus") {
      return svg("polygon", { points: [x, y - r, x + r, y, x, y + r, x - r, y].map(String).join(" ") });
    }
    // activity: triangle
    return svg("polygon", { points: [x, y - r, x + r, y + r, x - r, y + r].map(String).join(" ") });
  }

  // ── event wiring ─────────────────────────────────────────────────────────────
  function wireEpisode(g, ep) {
    g.addEventListener("mouseenter", function (e) {
      showTooltip(episodeTooltip(ep), e);
      if (state.selectedId == null) setEpisodeStrong(ep.id, true);
    });
    g.addEventListener("mousemove", positionTooltip);
    g.addEventListener("mouseleave", function () {
      hideTooltip();
      if (state.selectedId == null) setEpisodeStrong(ep.id, false);
    });
    g.addEventListener("click", function () { selectEpisode(ep.id); });
    g.addEventListener("focus", function () {
      if (state.selectedId == null) setEpisodeStrong(ep.id, true);
    });
    g.addEventListener("blur", function () {
      if (state.selectedId == null) setEpisodeStrong(ep.id, false);
    });
    g.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        selectEpisode(ep.id);
      }
    });
  }

  function wireGlyph(g, ep, link) {
    g.addEventListener("mouseenter", function (e) {
      showTooltip(linkTooltip(link), e);
      if (state.selectedId == null) setEpisodeStrong(ep.id, true);
    });
    g.addEventListener("mousemove", positionTooltip);
    g.addEventListener("mouseleave", function () {
      hideTooltip();
      if (state.selectedId == null) setEpisodeStrong(ep.id, false);
    });
    g.addEventListener("click", function () { selectEpisode(ep.id); });
  }

  // ── overview / brush ─────────────────────────────────────────────────────────
  function renderOverview() {
    overviewEl.innerHTML = "";
    const root = svg("svg", { viewBox: "0 0 " + W + " " + OV_H, class: "tl-ov-svg" });
    const trackY = 8;
    const trackH = OV_H - 22;
    root.appendChild(svg("rect", { x: PAD_L, y: trackY, width: PLOT_W, height: trackH, rx: 4, class: "tl-brush-track" }));

    state.data.nodes.forEach(function (ep) {
      if (ep.kind === "sensor_gap" ? !state.filters.sensor_gap : !state.filters[ep.kind]) return;
      const a = xFull(ms(ep.start));
      const b = xFull(ms(ep.end));
      const cls = ep.kind === "hyper" ? "tl-brush-tick-hyper" : ep.kind === "hypo" ? "tl-brush-tick-hypo" : "tl-brush-tick-gap";
      root.appendChild(svg("rect", { x: a, y: trackY + 2, width: Math.max(1.5, b - a), height: trackH - 4, class: cls, opacity: 0.75 }));
    });

    const wx0 = xFull(state.viewStart);
    const wx1 = xFull(state.viewEnd);
    const win = svg("rect", {
      x: wx0, y: trackY - 2, width: Math.max(6, wx1 - wx0), height: trackH + 4, rx: 3,
      class: "tl-brush-window", tabindex: 0, role: "slider",
      "aria-label": "Visible time range. Arrow keys pan, shift with arrows zoom.",
    });
    root.appendChild(win);
    root.appendChild(svg("rect", { x: wx0 - 2, y: trackY - 2, width: 4, height: trackH + 4, class: "tl-brush-handle", "data-handle": "start" }));
    root.appendChild(svg("rect", { x: wx1 - 2, y: trackY - 2, width: 4, height: trackH + 4, class: "tl-brush-handle", "data-handle": "end" }));

    overviewEl.appendChild(root);
    wireBrush(root, win);
  }

  function tForFullX(clientX, svgEl) {
    const rect = svgEl.getBoundingClientRect();
    const px = ((clientX - rect.left) / rect.width) * W;
    const frac = clamp((px - PAD_L) / PLOT_W, 0, 1);
    return state.fullStart + frac * (state.fullEnd - state.fullStart);
  }

  function wireBrush(svgEl, win) {
    let mode = null; // "move" | "start" | "end"
    let grabT = 0;

    function onDown(e) {
      const handle = e.target.getAttribute && e.target.getAttribute("data-handle");
      mode = handle || "move";
      grabT = tForFullX(e.clientX, svgEl);
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
      e.preventDefault();
    }
    function onMove(e) {
      const t = tForFullX(e.clientX, svgEl);
      if (mode === "start") {
        state.viewStart = clamp(t, state.fullStart, state.viewEnd - MIN_SPAN_MS);
      } else if (mode === "end") {
        state.viewEnd = clamp(t, state.viewStart + MIN_SPAN_MS, state.fullEnd);
      } else {
        const dt = t - grabT;
        grabT = t;
        panBy(dt);
      }
      render();
    }
    function onUp() {
      mode = null;
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    }
    svgEl.addEventListener("pointerdown", onDown);

    win.addEventListener("keydown", function (e) {
      const span = state.viewEnd - state.viewStart;
      const step = span * 0.15;
      if (e.key === "ArrowLeft" && e.shiftKey) { zoomBy(1.2); }
      else if (e.key === "ArrowRight" && e.shiftKey) { zoomBy(1 / 1.2); }
      else if (e.key === "ArrowLeft") { panBy(-step); render(); }
      else if (e.key === "ArrowRight") { panBy(step); render(); }
      else return;
      e.preventDefault();
    });
  }

  function panBy(dt) {
    const span = state.viewEnd - state.viewStart;
    let s = state.viewStart + dt;
    let en = state.viewEnd + dt;
    if (s < state.fullStart) { s = state.fullStart; en = s + span; }
    if (en > state.fullEnd) { en = state.fullEnd; s = en - span; }
    state.viewStart = s;
    state.viewEnd = en;
  }

  function zoomBy(factor, pivotT) {
    const span = state.viewEnd - state.viewStart;
    const fullSpan = state.fullEnd - state.fullStart;
    let newSpan = clamp(span * factor, MIN_SPAN_MS, fullSpan);
    const pivot = pivotT == null ? (state.viewStart + state.viewEnd) / 2 : pivotT;
    const leftFrac = (pivot - state.viewStart) / span;
    let s = pivot - leftFrac * newSpan;
    let en = s + newSpan;
    if (s < state.fullStart) { s = state.fullStart; en = s + newSpan; }
    if (en > state.fullEnd) { en = state.fullEnd; s = en - newSpan; }
    state.viewStart = s;
    state.viewEnd = en;
    render();
  }

  function resetView() {
    state.viewStart = state.fullStart;
    state.viewEnd = state.fullEnd;
    state.selectedId = null;
    render();
  }

  // ── controls ─────────────────────────────────────────────────────────────────
  function wireControls() {
    document.querySelectorAll("[data-filter]").forEach(function (cb) {
      cb.addEventListener("change", function () {
        state.filters[cb.dataset.filter] = cb.checked;
        render();
      });
    });
    if (resetBtn) resetBtn.addEventListener("click", resetView);

    chartEl.addEventListener("wheel", function (e) {
      e.preventDefault();
      const pivot = tForViewClientX(e.clientX);
      zoomBy(e.deltaY > 0 ? 1.15 : 1 / 1.15, pivot);
    }, { passive: false });
  }

  function tForViewClientX(clientX) {
    const svgEl = chartEl.querySelector("svg");
    if (!svgEl) return (state.viewStart + state.viewEnd) / 2;
    const rect = svgEl.getBoundingClientRect();
    const px = ((clientX - rect.left) / rect.width) * W;
    const frac = clamp((px - PAD_L) / PLOT_W, 0, 1);
    return state.viewStart + frac * (state.viewEnd - state.viewStart);
  }

  // ── boot ─────────────────────────────────────────────────────────────────────
  function boot(data) {
    state.data = data;
    const win = data.window || {};
    state.fullStart = ms(win.start) || 0;
    state.fullEnd = ms(win.end) || state.fullStart + 24 * 3600 * 1000;
    if (state.fullEnd <= state.fullStart) state.fullEnd = state.fullStart + 24 * 3600 * 1000;
    state.viewStart = state.fullStart;
    state.viewEnd = state.fullEnd;
    wireControls();
    render();
  }

  fetch("/episodes.json", { headers: { Accept: "application/json" } })
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (data) {
      if (!data || !Array.isArray(data.nodes)) {
        chartEl.innerHTML = '<p class="muted small">Could not load the episode graph.</p>';
        return;
      }
      boot(data);
    })
    .catch(function () {
      chartEl.innerHTML = '<p class="muted small">Could not load the episode graph.</p>';
    });
})();
