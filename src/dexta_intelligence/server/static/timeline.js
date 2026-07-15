// Temporal episode graph renderer for the Timeline page.
//
// Vanilla JS, inline SVG, no external libraries. Focus plus context: a compact
// navigator strip places every episode on the time axis (highs above a baseline,
// lows below, sensor gaps as ticks), and the large focus view draws the actual
// glucose curve around the selected episode with each meal, bolus, activity, and
// sleep placed at its real time, connected to the episode, and labelled with the
// human relation ("meal 45g, 27 min before"). Depth (gradients, soft shadow, a
// glowing extreme marker) is pure presentation; every number comes from the
// server. Nothing here is authored by a model.

(function () {
  "use strict";

  const shellEl = document.querySelector(".tl-shell");
  if (!shellEl) return;
  const navEl = document.getElementById("tl-navigator");
  const plotEl = document.getElementById("tl-plot");
  const detailEl = document.getElementById("tl-detail");
  const tooltipEl = document.getElementById("tl-tooltip");
  const titleEl = document.getElementById("tl-focus-title");
  const posEl = document.getElementById("tl-position");
  const prevBtn = document.getElementById("tl-prev");
  const nextBtn = document.getElementById("tl-next");
  if (!navEl || !plotEl) return;

  const SVGNS = "http://www.w3.org/2000/svg";
  const KIND_TITLE = { hypo: "Low episode", hyper: "High episode", sensor_gap: "Sensor gap" };
  const EXTREME_NOUN = { hypo: "nadir", hyper: "peak" };
  const GAP_BREAK_MS = 20 * 60 * 1000;

  const state = {
    data: null,
    fullStart: 0,
    fullEnd: 0,
    target: { low: 70, high: 180 },
    selectedId: null,
    detail: null,
    view: "curve",
    filters: { hypo: true, hyper: true, sensor_gap: true },
  };

  // ── small helpers ────────────────────────────────────────────────────────────
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

  function num(v) {
    return typeof v === "number" && isFinite(v);
  }

  function trimNum(v) {
    return num(v) ? String(Math.round(v * 100) / 100) : "";
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

  // ── shared text: kind detail and human relation ──────────────────────────────
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

  // Compact glyph label, e.g. "meal 45g" / "bolus 5.2U" / "activity run".
  function glyphLabel(kind, detail) {
    detail = detail || {};
    if (kind === "meal") return num(detail.carbs_g) ? "meal " + trimNum(detail.carbs_g) + "g" : "meal";
    if (kind === "bolus") return num(detail.units) ? "bolus " + trimNum(detail.units) + "U" : "bolus";
    if (kind === "activity") return detail.kind ? "activity " + String(detail.kind) : "activity";
    if (kind === "sleep") return num(detail.score) ? "sleep score " + trimNum(detail.score) : "sleep";
    return kind;
  }

  // Signed offset (minutes from episode start) as a plain-language relation.
  function relationWord(offsetMin) {
    if (!num(offsetMin)) return "";
    const a = Math.abs(offsetMin);
    if (a < 1) return "at onset";
    const mag = a >= 90 ? (a / 60).toFixed(1) + " h" : Math.round(a) + " min";
    return mag + (offsetMin < 0 ? " before" : " after");
  }

  // ── tooltip ──────────────────────────────────────────────────────────────────
  function showTooltip(html, evt) {
    if (!tooltipEl) return;
    tooltipEl.innerHTML = html;
    tooltipEl.hidden = false;
    positionTooltip(evt);
  }

  function positionTooltip(evt) {
    if (!tooltipEl || tooltipEl.hidden) return;
    const rect = shellEl.getBoundingClientRect();
    let x = evt.clientX - rect.left + 14;
    const y = evt.clientY - rect.top + 14;
    const w = tooltipEl.offsetWidth;
    if (x + w > rect.width) x = rect.width - w - 4;
    tooltipEl.style.left = Math.max(0, x) + "px";
    tooltipEl.style.top = Math.max(0, y) + "px";
  }

  function hideTooltip() {
    if (tooltipEl) tooltipEl.hidden = true;
  }

  // Depth defs shared by both surfaces: soft drop shadow, marker glow, and the
  // curve / target gradients. Presentation only.
  function appendDefs(root, idns) {
    const defs = svg("defs");

    const shadow = svg("filter", { id: idns + "-shadow", x: "-20%", y: "-20%", width: "140%", height: "160%" });
    shadow.appendChild(svg("feDropShadow", { dx: 0, dy: 1.4, stdDeviation: 1.6, "flood-color": "rgba(15,23,42,0.35)" }));
    defs.appendChild(shadow);

    const glow = svg("filter", { id: idns + "-glow", x: "-80%", y: "-80%", width: "260%", height: "260%" });
    glow.appendChild(svg("feGaussianBlur", { stdDeviation: 3.4, result: "b" }));
    const merge = svg("feMerge");
    merge.appendChild(svg("feMergeNode", { in: "b" }));
    merge.appendChild(svg("feMergeNode", { in: "SourceGraphic" }));
    glow.appendChild(merge);
    defs.appendChild(glow);

    ["hyper", "hypo"].forEach(function (kind) {
      const grad = svg("linearGradient", { id: idns + "-fill-" + kind, x1: 0, y1: 0, x2: 0, y2: 1 });
      grad.appendChild(svg("stop", { offset: "0%", class: "tl-grad-" + kind + "-a" }));
      grad.appendChild(svg("stop", { offset: "100%", class: "tl-grad-" + kind + "-b" }));
      defs.appendChild(grad);
    });

    const target = svg("linearGradient", { id: idns + "-target", x1: 0, y1: 0, x2: 0, y2: 1 });
    target.appendChild(svg("stop", { offset: "0%", class: "tl-grad-target-a" }));
    target.appendChild(svg("stop", { offset: "100%", class: "tl-grad-target-b" }));
    defs.appendChild(target);

    const stroke = svg("linearGradient", { id: idns + "-curve", x1: 0, y1: 0, x2: 1, y2: 0 });
    stroke.appendChild(svg("stop", { offset: "0%", class: "tl-grad-curve-a" }));
    stroke.appendChild(svg("stop", { offset: "100%", class: "tl-grad-curve-b" }));
    defs.appendChild(stroke);

    root.appendChild(defs);
  }

  // ── ordered / filtered episode list (drives navigator and prev/next) ──────────
  function orderedEpisodes() {
    return state.data.nodes.filter(function (n) {
      const key = n.kind === "sensor_gap" ? "sensor_gap" : n.kind;
      return state.filters[key];
    });
  }

  // ── navigator strip ───────────────────────────────────────────────────────────
  const NV_W = 960;
  const NV_PAD_L = 40;
  const NV_PAD_R = 16;
  const NV_PLOT_W = NV_W - NV_PAD_L - NV_PAD_R;
  const NV_BASE_Y = 46;
  const NV_H = 92;

  function xFull(t) {
    const span = state.fullEnd - state.fullStart || 1;
    return NV_PAD_L + ((t - state.fullStart) / span) * NV_PLOT_W;
  }

  function markHeight(ep) {
    if (ep.severe) return 30;
    if (ep.clinically_significant) return 22;
    return 14;
  }

  function renderNavigator() {
    navEl.innerHTML = "";
    const root = svg("svg", { viewBox: "0 0 " + NV_W + " " + NV_H, class: "tl-nv-svg", "aria-hidden": "true" });
    appendDefs(root, "nv");

    root.appendChild(svg("line", { x1: NV_PAD_L, y1: NV_BASE_Y, x2: NV_W - NV_PAD_R, y2: NV_BASE_Y, class: "tl-nv-base" }));
    root.appendChild(navSideLabel("Highs", NV_BASE_Y - 30));
    root.appendChild(navSideLabel("Lows", NV_BASE_Y + 34));

    navTicks().forEach(function (t) {
      const x = xFull(t);
      const label = svg("text", { x: x, y: NV_H - 4, class: "tl-nv-tick", "text-anchor": "middle" });
      label.textContent = navSpansDays() ? fmtDay(t) : fmtTime(t);
      root.appendChild(label);
    });

    orderedEpisodes().forEach(function (ep) {
      root.appendChild(navMark(ep));
    });
    navEl.appendChild(root);
  }

  function navSideLabel(text, y) {
    const t = svg("text", { x: NV_PAD_L - 6, y: y, class: "tl-nv-side", "text-anchor": "end" });
    t.textContent = text;
    return t;
  }

  function navSpansDays() {
    return state.fullEnd - state.fullStart > 2 * 24 * 3600 * 1000;
  }

  function navTicks() {
    const n = 6;
    const out = [];
    for (let i = 0; i <= n; i++) out.push(state.fullStart + ((state.fullEnd - state.fullStart) * i) / n);
    return out;
  }

  function navMark(ep) {
    const a = ms(ep.start);
    const b = ms(ep.end);
    const x = clamp((xFull(a) + xFull(b)) / 2, NV_PAD_L, NV_W - NV_PAD_R);
    const sel = ep.id === state.selectedId;
    const cls = "tl-nv-mark tl-nv-" + ep.kind + (ep.severe ? " tl-nv-severe" : "") + (sel ? " tl-nv-sel" : "");
    const g = svg("g", { class: cls, tabindex: 0, role: "button", "data-ep": ep.id });
    g.setAttribute("aria-label", navAria(ep));
    g.setAttribute("aria-pressed", sel ? "true" : "false");

    if (ep.kind === "sensor_gap") {
      const x0 = clamp(xFull(a), NV_PAD_L, NV_W - NV_PAD_R);
      const x1 = clamp(xFull(b), NV_PAD_L, NV_W - NV_PAD_R);
      g.appendChild(svg("rect", { x: x0, y: NV_BASE_Y - 26, width: Math.max(2, x1 - x0), height: 52, class: "tl-nv-gap-fill" }));
    } else {
      const h = markHeight(ep);
      const up = ep.kind === "hyper";
      const y2 = up ? NV_BASE_Y - h : NV_BASE_Y + h;
      g.appendChild(svg("line", { x1: x, y1: NV_BASE_Y, x2: x, y2: y2, class: "tl-nv-stem" }));
      const r = ep.severe ? 5 : 4;
      const dot = svg("circle", { cx: x, cy: y2, r: r, class: "tl-nv-dot", filter: "url(#nv-shadow)" });
      g.appendChild(dot);
      if (ep.severe) {
        const mark = svg("text", { x: x, y: up ? y2 - 8 : y2 + 14, class: "tl-nv-bang", "text-anchor": "middle" });
        mark.textContent = "!";
        g.appendChild(mark);
      }
    }
    wireMark(g, ep);
    return g;
  }

  function navAria(ep) {
    if (ep.kind === "sensor_gap") {
      return "Sensor gap, " + trimNum(ep.duration_min) + " minutes from " + fmtStamp(ms(ep.start)) + " UTC. Select to view.";
    }
    const kind = ep.kind === "hyper" ? "high" : "low";
    const sev = ep.severe ? "severe " : ep.clinically_significant ? "clinically significant " : "";
    return sev + kind + " episode, " + trimNum(ep.extreme_mg_dl) + " milligrams per deciliter, "
      + trimNum(ep.duration_min) + " minutes from " + fmtStamp(ms(ep.start)) + " UTC. Select to view relations.";
  }

  function wireMark(g, ep) {
    g.addEventListener("click", function () { select(ep.id); });
    g.addEventListener("mouseenter", function (e) { showTooltip(navTooltip(ep), e); });
    g.addEventListener("mousemove", positionTooltip);
    g.addEventListener("mouseleave", hideTooltip);
    g.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        select(ep.id);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        step(1, true);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        step(-1, true);
      }
    });
  }

  function navTooltip(ep) {
    const rows = ['<div class="tl-tt-title">' + (KIND_TITLE[ep.kind] || "Episode") + "</div>"];
    rows.push('<div class="tl-tt-row">' + fmtStamp(ms(ep.start)) + " UTC</div>");
    if (num(ep.duration_min)) rows.push('<div class="tl-tt-row">' + trimNum(ep.duration_min) + " min</div>");
    if (num(ep.extreme_mg_dl)) {
      rows.push('<div class="tl-tt-row">' + trimNum(ep.extreme_mg_dl) + " mg/dL " + (EXTREME_NOUN[ep.kind] || "extreme") + "</div>");
    }
    return rows.join("");
  }

  // ── focus plot geometry ────────────────────────────────────────────────────────
  const FW = 960;
  const F_PAD_L = 48;
  const F_PAD_R = 18;
  const F_PLOT_W = FW - F_PAD_L - F_PAD_R;
  const F_ROW_H = 32;
  const F_BAND_TOP = 10;
  const F_PLOT_H = 214;

  function fx(t, win) {
    const span = win.end - win.start || 1;
    return F_PAD_L + ((t - win.start) / span) * F_PLOT_W;
  }

  // ── focus render: dispatch to the active lens ─────────────────────────────────
  function renderFocus(detail) {
    if (state.view === "graph") renderGraph(detail);
    else renderCurve(detail);
  }

  function renderCurve(detail) {
    plotEl.innerHTML = "";
    plotEl.setAttribute("aria-label", "Glucose curve around the selected episode, with each"
      + " meal, bolus, activity, and sleep placed at its real time and labelled with how it"
      + " relates to the episode.");
    const ep = detail.episode;
    const win = { start: ms(detail.window.start), end: ms(detail.window.end) };
    const series = (detail.series || []).map(function (r) { return { t: ms(r.t), v: r.v }; })
      .filter(function (r) { return r.t != null && num(r.v); });

    const events = layoutEvents(ep, win);
    const nRows = events.reduce(function (m, e) { return Math.max(m, e.row + 1); }, 0);
    const bandH = Math.max(1, nRows) * F_ROW_H;
    const plotTop = F_BAND_TOP + bandH + 8;
    const plotBot = plotTop + F_PLOT_H;
    const axisY = plotBot + 4;
    const H = axisY + 24;
    const fillKind = ep.kind === "hypo" ? "hypo" : "hyper";

    const yDomain = glucoseDomain(series, ep);
    function gy(v) {
      const frac = (v - yDomain.lo) / (yDomain.hi - yDomain.lo || 1);
      return plotBot - frac * (plotBot - plotTop);
    }

    const root = svg("svg", { viewBox: "0 0 " + FW + " " + H, class: "tl-f-svg", "aria-hidden": "true" });
    appendDefs(root, "fx");

    // target band + gridlines
    const yHi = gy(state.target.high);
    const yLo = gy(state.target.low);
    root.appendChild(svg("rect", { x: F_PAD_L, y: yHi, width: F_PLOT_W, height: Math.max(0, yLo - yHi), class: "tl-f-target", fill: "url(#fx-target)" }));
    [state.target.low, state.target.high].forEach(function (v) {
      const y = gy(v);
      root.appendChild(svg("line", { x1: F_PAD_L, y1: y, x2: FW - F_PAD_R, y2: y, class: "tl-f-grid" }));
      const lab = svg("text", { x: F_PAD_L - 6, y: y + 3, class: "tl-f-yaxis", "text-anchor": "end" });
      lab.textContent = String(v);
      root.appendChild(lab);
    });

    // episode region
    const rx0 = clamp(fx(ms(ep.start), win), F_PAD_L, FW - F_PAD_R);
    const rx1 = clamp(fx(ms(ep.end), win), F_PAD_L, FW - F_PAD_R);
    root.appendChild(svg("rect", {
      x: rx0, y: plotTop, width: Math.max(2, rx1 - rx0), height: plotBot - plotTop,
      class: "tl-f-region tl-f-region-" + ep.kind,
    }));

    // glucose curve, with a gradient area fill beneath it for depth
    drawCurve(root, series, win, gy, plotBot, fillKind);

    // x axis
    root.appendChild(svg("line", { x1: F_PAD_L, y1: axisY, x2: FW - F_PAD_R, y2: axisY, class: "tl-f-grid" }));
    focusTicks(win).forEach(function (t) {
      const x = fx(t, win);
      const lab = svg("text", { x: x, y: axisY + 14, class: "tl-f-xaxis", "text-anchor": "middle" });
      lab.textContent = fmtTime(t);
      root.appendChild(lab);
    });

    // extreme marker (glowing)
    let extreme = null;
    if (num(ep.extreme_mg_dl) && ep.extreme_ts) {
      const ext = ms(ep.extreme_ts);
      const ex = clamp(fx(ext, win), F_PAD_L, FW - F_PAD_R);
      const ey = gy(ep.extreme_mg_dl);
      extreme = { x: ex, y: ey };
      root.appendChild(svg("circle", { cx: ex, cy: ey, r: 7, class: "tl-f-extreme tl-f-extreme-" + ep.kind, filter: "url(#fx-glow)" }));
      root.appendChild(svg("circle", { cx: ex, cy: ey, r: 2.6, class: "tl-f-extreme-core" }));
      const up = ep.kind === "hyper";
      const lab = svg("text", { x: ex, y: up ? ey - 12 : ey + 19, class: "tl-f-extreme-label", "text-anchor": "middle" });
      lab.textContent = (EXTREME_NOUN[ep.kind] || "extreme") + " " + trimNum(ep.extreme_mg_dl) + " mg/dL";
      root.appendChild(lab);
    }

    // connectors from each event to the episode, then the labelled glyphs
    const target = extreme || { x: (rx0 + rx1) / 2, y: plotTop };
    events.forEach(function (ev) {
      drawConnector(root, ev, target, plotBot);
    });
    events.forEach(function (ev) {
      drawEventGlyph(root, ev);
    });

    if (!series.length) {
      const msg = svg("text", { x: FW / 2, y: (plotTop + plotBot) / 2, class: "tl-f-empty", "text-anchor": "middle" });
      msg.textContent = "No glucose readings in this window";
      root.appendChild(msg);
    }

    plotEl.appendChild(root);
  }

  function glucoseDomain(series, ep) {
    let lo = state.target.low - 15;
    let hi = state.target.high + 15;
    series.forEach(function (r) {
      if (r.v < lo) lo = r.v;
      if (r.v > hi) hi = r.v;
    });
    if (num(ep.extreme_mg_dl)) {
      if (ep.extreme_mg_dl < lo) lo = ep.extreme_mg_dl;
      if (ep.extreme_mg_dl > hi) hi = ep.extreme_mg_dl;
    }
    const pad = Math.max(8, (hi - lo) * 0.08);
    return { lo: Math.max(0, lo - pad), hi: hi + pad };
  }

  // Curve as broken segments (across sensor gaps); each segment also gets a
  // gradient area fill down to the plot floor for a sense of depth.
  function drawCurve(root, series, win, gy, plotBot, fillKind) {
    if (!series.length) return;
    let seg = [];
    let prev = null;
    const flush = function () {
      if (seg.length < 1) { seg = []; return; }
      let line = "";
      seg.forEach(function (p, i) {
        line += (i === 0 ? "M" : "L") + p[0].toFixed(1) + " " + p[1].toFixed(1);
      });
      if (seg.length > 1) {
        const area = "M" + seg[0][0].toFixed(1) + " " + plotBot.toFixed(1) + line.slice(1)
          + "L" + seg[seg.length - 1][0].toFixed(1) + " " + plotBot.toFixed(1) + "Z";
        root.appendChild(svg("path", { d: area, class: "tl-f-area", fill: "url(#fx-fill-" + fillKind + ")" }));
      }
      root.appendChild(svg("path", { d: line, class: "tl-f-curve", stroke: "url(#fx-curve)" }));
      seg = [];
    };
    series.forEach(function (r) {
      const x = fx(r.t, win);
      const y = gy(r.v);
      if (prev != null && r.t - prev > GAP_BREAK_MS) flush();
      seg.push([x, y]);
      prev = r.t;
    });
    flush();
  }

  function focusTicks(win) {
    const n = 6;
    const out = [];
    for (let i = 0; i <= n; i++) out.push(win.start + ((win.end - win.start) * i) / n);
    return out;
  }

  // Place event glyphs and labels in the top band, packed into rows so nothing
  // overlaps. Returns geometry the connector and glyph drawers consume.
  function layoutEvents(ep, win) {
    const links = (ep.links || []).slice().sort(function (a, b) {
      return (ms(a.ts) || 0) - (ms(b.ts) || 0);
    });
    const rows = [];
    return links.map(function (link) {
      const t = ms(link.ts);
      const x = clamp(fx(t, win), F_PAD_L + 4, FW - F_PAD_R - 4);
      const line1 = glyphLabel(link.kind, link.detail);
      const line2 = relationWord(link.offset_min);
      const textW = Math.max(line1.length, line2.length) * 6.0;
      const boxW = 20 + textW;
      const anchorRight = x + boxW > FW - F_PAD_R;
      const lo = anchorRight ? x - boxW : x;
      const hi = anchorRight ? x : x + boxW;
      let row = 0;
      while (row < rows.length && overlaps(rows[row], lo, hi)) row++;
      if (row === rows.length) rows.push([]);
      rows[row].push([lo - 6, hi + 6]);
      return {
        link: link, x: x, row: row, anchorRight: anchorRight,
        y: F_BAND_TOP + row * F_ROW_H + 14, line1: line1, line2: line2, ep: ep,
      };
    });
  }

  function overlaps(intervals, lo, hi) {
    for (let i = 0; i < intervals.length; i++) {
      if (lo < intervals[i][1] && hi > intervals[i][0]) return true;
    }
    return false;
  }

  function drawConnector(root, ev, target, plotBot) {
    const gx = ev.x;
    const gy0 = ev.y + 6;
    const cx = (gx + target.x) / 2;
    const cy = (plotBot + target.y) / 2;
    const d = "M" + gx.toFixed(1) + " " + gy0.toFixed(1)
      + " C" + gx.toFixed(1) + " " + cy.toFixed(1)
      + " " + cx.toFixed(1) + " " + target.y.toFixed(1)
      + " " + target.x.toFixed(1) + " " + target.y.toFixed(1);
    root.appendChild(svg("path", { d: d, class: "tl-f-conn tl-f-conn-" + ev.link.kind }));
  }

  function drawEventGlyph(root, ev) {
    const link = ev.link;
    const g = svg("g", { class: "tl-f-event tl-f-event-" + link.kind, tabindex: 0, role: "img" });
    g.setAttribute("aria-label", eventAria(link));
    const shape = glyphShape(link.kind, ev.x, ev.y);
    shape.setAttribute("filter", "url(#fx-shadow)");
    g.appendChild(shape);
    const tx = ev.anchorRight ? ev.x - 10 : ev.x + 10;
    const anchor = ev.anchorRight ? "end" : "start";
    const l1 = svg("text", { x: tx, y: ev.y - 1, class: "tl-f-lab1", "text-anchor": anchor });
    l1.textContent = ev.line1;
    g.appendChild(l1);
    if (ev.line2) {
      const l2 = svg("text", { x: tx, y: ev.y + 12, class: "tl-f-lab2", "text-anchor": anchor });
      l2.textContent = ev.line2;
      g.appendChild(l2);
    }
    g.addEventListener("mouseenter", function (e) { showTooltip(eventTooltip(link), e); });
    g.addEventListener("mousemove", positionTooltip);
    g.addEventListener("mouseleave", hideTooltip);
    root.appendChild(g);
  }

  function glyphShape(kind, x, y) {
    const r = 5;
    if (kind === "meal") return svg("circle", { cx: x, cy: y, r: r, class: "tl-f-shape" });
    if (kind === "sleep") return svg("rect", { x: x - r, y: y - r, width: 2 * r, height: 2 * r, class: "tl-f-shape" });
    if (kind === "bolus") return svg("polygon", { points: [x, y - r, x + r, y, x, y + r, x - r, y].map(String).join(" "), class: "tl-f-shape" });
    return svg("polygon", { points: [x, y - r, x + r, y + r, x - r, y + r].map(String).join(" "), class: "tl-f-shape" });
  }

  function eventAria(link) {
    const d = edgeDetail(link.kind, link.detail);
    const rel = relationWord(link.offset_min);
    return link.kind + (d ? ", " + d : "") + (rel ? ", " + rel : "") + ", at " + fmtTime(ms(link.ts)) + " UTC.";
  }

  function eventTooltip(link) {
    const rows = ['<div class="tl-tt-title">' + link.kind + "</div>"];
    const d = edgeDetail(link.kind, link.detail);
    if (d) rows.push('<div class="tl-tt-row">' + d + "</div>");
    const rel = relationWord(link.offset_min);
    rows.push('<div class="tl-tt-row">' + fmtTime(ms(link.ts)) + " UTC" + (rel ? " · " + rel : "") + "</div>");
    return rows.join("");
  }

  // ── ego-graph lens ─────────────────────────────────────────────────────────────
  // The selected episode as a centre node with its typed context events as
  // satellites. Layout is deterministic: side by the sign of offset_min (before
  // left, after right), angular sector by kind, radius by |offset_min|. No
  // physics, no jitter; the same links always yield the same picture.
  const G_KIND_SLOT = { meal: -45, bolus: -15, activity: 15, sleep: 45 };
  const G_W = 960;
  const G_H = 620;
  const G_R_MIN = 140;
  const G_R_MAX = 250;

  function graphAngleDeg(kind, side) {
    const base = G_KIND_SLOT[kind] != null ? G_KIND_SLOT[kind] : 0;
    return side === "before" ? 180 - base : base;
  }

  function layoutGraphNodes(links) {
    const cx = G_W / 2;
    const cy = G_H / 2;
    const maxAbs = links.reduce(function (m, l) {
      return num(l.offset_min) ? Math.max(m, Math.abs(l.offset_min)) : m;
    }, 1);
    const groups = {};
    links.forEach(function (l) {
      const side = num(l.offset_min) && l.offset_min < 0 ? "before" : "after";
      const key = side + ":" + l.kind;
      (groups[key] = groups[key] || []).push(l);
    });
    const out = [];
    Object.keys(groups).sort().forEach(function (key) {
      const arr = groups[key].slice().sort(function (a, b) {
        return (a.offset_min || 0) - (b.offset_min || 0);
      });
      const side = key.indexOf("before:") === 0 ? "before" : "after";
      const kind = key.slice(side.length + 1);
      const base = graphAngleDeg(kind, side);
      const n = arr.length;
      arr.forEach(function (link, i) {
        const a = (base + (i - (n - 1) / 2) * 9) * Math.PI / 180;
        const off = num(link.offset_min) ? Math.abs(link.offset_min) : 0;
        const r = G_R_MIN + (off / maxAbs) * (G_R_MAX - G_R_MIN);
        out.push({ link: link, x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) });
      });
    });
    return out;
  }

  function renderGraph(detail) {
    plotEl.innerHTML = "";
    plotEl.setAttribute("aria-label", "Node-link ego graph: the selected episode at the centre"
      + " with each linked meal, bolus, activity, and sleep as a satellite node, and edges"
      + " labelled with the time relation to the episode.");
    const ep = detail.episode;
    const cx = G_W / 2;
    const cy = G_H / 2;
    const links = ep.links || [];

    const root = svg("svg", { viewBox: "0 0 " + G_W + " " + G_H, class: "tl-g-svg", "aria-hidden": "true" });
    appendDefs(root, "g");

    const nodes = layoutGraphNodes(links);
    nodes.forEach(function (p) { drawGraphEdge(root, p, cx, cy); });
    drawCenterNode(root, ep, cx, cy);
    nodes.forEach(function (p) { drawGraphNode(root, p, cx); });

    if (!links.length) {
      const note = svg("text", { x: cx, y: cy + 96, class: "tl-g-note", "text-anchor": "middle" });
      note.textContent = ep.kind === "sensor_gap"
        ? "Sensor gaps have no linked context."
        : "No meals, boluses, activity, or sleep were linked to this episode.";
      root.appendChild(note);
    }
    plotEl.appendChild(root);
  }

  function centerSub(ep) {
    if (ep.kind === "sensor_gap") {
      return num(ep.duration_min) ? trimNum(ep.duration_min) + " min gap" : "sensor gap";
    }
    const parts = [];
    if (num(ep.extreme_mg_dl)) parts.push(trimNum(ep.extreme_mg_dl) + " mg/dL " + (EXTREME_NOUN[ep.kind] || "extreme"));
    if (num(ep.duration_min)) parts.push(trimNum(ep.duration_min) + " min");
    return parts.join(" · ");
  }

  function centerAria(ep) {
    const sev = ep.severe ? ", severe" : ep.clinically_significant ? ", clinically significant" : "";
    return (KIND_TITLE[ep.kind] || "Episode") + ", " + centerSub(ep) + sev + ". Centre of the graph.";
  }

  function drawCenterNode(root, ep, cx, cy) {
    const cw = 184;
    const ch = 78;
    const sevClass = ep.severe ? " tl-g-severe" : ep.clinically_significant ? " tl-g-sig" : "";
    const g = svg("g", { class: "tl-g-center tl-g-center-" + ep.kind + sevClass, tabindex: 0, role: "img" });
    g.setAttribute("aria-label", centerAria(ep));
    g.appendChild(svg("rect", { x: cx - cw / 2, y: cy - ch / 2, width: cw, height: ch, rx: 12, class: "tl-g-center-box", filter: "url(#g-shadow)" }));
    const title = svg("text", { x: cx, y: cy - 16, class: "tl-g-center-title", "text-anchor": "middle" });
    title.textContent = KIND_TITLE[ep.kind] || "Episode";
    g.appendChild(title);
    const sub = svg("text", { x: cx, y: cy + 4, class: "tl-g-center-sub", "text-anchor": "middle" });
    sub.textContent = centerSub(ep);
    g.appendChild(sub);
    if (ep.severe || ep.clinically_significant) {
      const badge = svg("text", { x: cx, y: cy + 24, class: "tl-g-center-badge", "text-anchor": "middle" });
      badge.textContent = ep.severe ? "severe" : "clinically significant";
      g.appendChild(badge);
    }
    root.appendChild(g);
  }

  function drawGraphEdge(root, p, cx, cy) {
    const link = p.link;
    root.appendChild(svg("line", {
      x1: cx, y1: cy, x2: p.x.toFixed(1), y2: p.y.toFixed(1),
      class: "tl-g-edge tl-g-edge-" + link.kind,
    }));
    const rel = relationWord(link.offset_min);
    if (rel) {
      const lx = cx + (p.x - cx) * 0.58;
      const ly = cy + (p.y - cy) * 0.58;
      const t = svg("text", { x: lx.toFixed(1), y: ly.toFixed(1), class: "tl-g-edge-label", "text-anchor": "middle" });
      t.textContent = rel;
      root.appendChild(t);
    }
  }

  function drawGraphNode(root, p, cx) {
    const link = p.link;
    const g = svg("g", { class: "tl-f-event tl-g-node tl-f-event-" + link.kind, tabindex: 0, role: "img" });
    g.setAttribute("aria-label", eventAria(link));
    g.appendChild(svg("circle", { cx: p.x.toFixed(1), cy: p.y.toFixed(1), r: 11, class: "tl-g-node-halo" }));
    const shape = glyphShape(link.kind, p.x, p.y);
    shape.setAttribute("filter", "url(#g-shadow)");
    g.appendChild(shape);
    const left = p.x < cx;
    const tx = left ? p.x - 16 : p.x + 16;
    const lab = svg("text", { x: tx.toFixed(1), y: (p.y + 4).toFixed(1), class: "tl-g-node-label", "text-anchor": left ? "end" : "start" });
    lab.textContent = glyphLabel(link.kind, link.detail);
    g.appendChild(lab);
    g.addEventListener("mouseenter", function (e) { showTooltip(eventTooltip(link), e); });
    g.addEventListener("mousemove", positionTooltip);
    g.addEventListener("mouseleave", hideTooltip);
    root.appendChild(g);
  }

  // ── facts card (mirrors _episode_card.html) ────────────────────────────────────
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
    if (!detailEl) return;
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

  // ── selection + stepping ──────────────────────────────────────────────────────
  function nodeById(id) {
    return state.data.nodes.find(function (n) { return n.id === id; }) || null;
  }

  function updateHeader(ep) {
    if (titleEl) {
      const badge = ep.severe ? " (severe)" : ep.clinically_significant ? " (clinically significant)" : "";
      titleEl.textContent = (KIND_TITLE[ep.kind] || "Episode") + " · " + fmtStamp(ms(ep.start)) + badge;
    }
    if (posEl) {
      const list = orderedEpisodes();
      const idx = list.findIndex(function (n) { return n.id === ep.id; });
      posEl.textContent = idx >= 0 ? (idx + 1) + " of " + list.length : "";
    }
  }

  function select(id) {
    if (id == null) return;
    state.selectedId = id;
    renderNavigator();
    const ep = nodeById(id);
    if (!ep) return;
    renderDetail(ep);
    updateHeader(ep);
    fetchDetail(id);
  }

  function fetchDetail(id) {
    plotEl.classList.add("tl-loading");
    fetch("/episode.json?id=" + encodeURIComponent(id), { headers: { Accept: "application/json" } })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (detail) {
        plotEl.classList.remove("tl-loading");
        if (state.selectedId !== id) return;
        if (!detail || !detail.episode) {
          showDetailError();
          return;
        }
        state.detail = detail;
        renderFocus(detail);
      })
      .catch(function () {
        plotEl.classList.remove("tl-loading");
        if (state.selectedId === id) showDetailError();
      });
  }

  // A failed fetch must never leave a previous episode's detail around: a lens
  // toggle would silently re-render it under the newly selected header.
  function showDetailError() {
    state.detail = null;
    plotEl.innerHTML = '<p class="muted small">Could not load this episode.'
      + " The episode list may be out of date; reload the page.</p>";
  }

  function step(dir, wrap) {
    const list = orderedEpisodes();
    if (!list.length) return;
    let idx = list.findIndex(function (n) { return n.id === state.selectedId; });
    if (idx < 0) {
      select(list[dir > 0 ? 0 : list.length - 1].id);
      return;
    }
    idx += dir;
    if (idx < 0) idx = wrap ? list.length - 1 : 0;
    if (idx >= list.length) idx = wrap ? 0 : list.length - 1;
    select(list[idx].id);
  }

  // ── controls ──────────────────────────────────────────────────────────────────
  function wireControls() {
    document.querySelectorAll("[data-filter]").forEach(function (cb) {
      cb.addEventListener("change", function () {
        state.filters[cb.dataset.filter] = cb.checked;
        renderNavigator();
        const list = orderedEpisodes();
        if (list.length && !list.some(function (n) { return n.id === state.selectedId; })) {
          select(list[0].id);
        } else if (state.selectedId) {
          updateHeader(nodeById(state.selectedId));
        }
      });
    });
    if (prevBtn) prevBtn.addEventListener("click", function () { step(-1, true); });
    if (nextBtn) nextBtn.addEventListener("click", function () { step(1, true); });
    document.querySelectorAll(".tl-vt-btn").forEach(function (btn) {
      btn.addEventListener("click", function () { setView(btn.dataset.view); });
    });
  }

  function setView(view) {
    if (view !== "curve" && view !== "graph") return;
    if (state.view === view) return;
    state.view = view;
    updateViewToggle();
    if (state.detail && state.detail.episode && state.detail.episode.id === state.selectedId) {
      renderFocus(state.detail);
    } else if (state.selectedId) {
      fetchDetail(state.selectedId);
    }
  }

  function updateViewToggle() {
    document.querySelectorAll(".tl-vt-btn").forEach(function (btn) {
      const on = btn.dataset.view === state.view;
      btn.classList.toggle("on", on);
      btn.setAttribute("aria-pressed", on ? "true" : "false");
    });
  }

  // ── boot ──────────────────────────────────────────────────────────────────────
  function salientId() {
    const attr = shellEl.getAttribute("data-default-episode");
    if (attr && nodeById(attr)) return attr;
    const excursions = state.data.nodes.filter(function (n) { return n.kind === "hypo" || n.kind === "hyper"; });
    if (!excursions.length) return state.data.nodes.length ? state.data.nodes[0].id : null;
    excursions.sort(function (a, b) {
      const sa = (a.severe ? 4 : 0) + (a.clinically_significant ? 2 : 0);
      const sb = (b.severe ? 4 : 0) + (b.clinically_significant ? 2 : 0);
      if (sa !== sb) return sb - sa;
      return (b.duration_min || 0) - (a.duration_min || 0);
    });
    return excursions[0].id;
  }

  function boot(data) {
    state.data = data;
    const win = data.window || {};
    state.fullStart = ms(win.start) || 0;
    state.fullEnd = ms(win.end) || state.fullStart + 24 * 3600 * 1000;
    if (state.fullEnd <= state.fullStart) state.fullEnd = state.fullStart + 24 * 3600 * 1000;
    if (data.target) state.target = data.target;
    wireControls();
    renderNavigator();
    const id = salientId();
    if (id) select(id);
  }

  fetch("/episodes.json", { headers: { Accept: "application/json" } })
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (data) {
      if (!data || !Array.isArray(data.nodes)) {
        navEl.innerHTML = '<p class="muted small">Could not load the episode graph.</p>';
        return;
      }
      boot(data);
    })
    .catch(function () {
      navEl.innerHTML = '<p class="muted small">Could not load the episode graph.</p>';
    });
})();
