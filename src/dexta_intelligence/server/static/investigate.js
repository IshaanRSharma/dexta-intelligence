// Live investigation stream for the Investigations page.
//
// Two modes against /api/investigate/stream:
//   question (default) - the orchestrator drills the question over the tool belt,
//     emitting per-tool tool_call/tool_result events (the live tool shelf) and a
//     final audited answer. This is the PRD's plan -> trace -> evidence flow.
//   deep - the coordinator runs the multi-producer statistical sweep, emitting
//     plan/running/producer_done and a final set of evidence cards.

(function () {
  "use strict";

  const form = document.getElementById("investigate-form");
  const input = document.getElementById("investigate-q");
  const root = document.getElementById("investigate-stream");
  const deepBtn = document.getElementById("investigate-deep");
  if (!form || !input || !root) return;

  let active = null; // in-flight EventSource

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  function toolLabel(name) {
    return String(name || "tool").replace(/_/g, " ");
  }

  function traceIcon(icon) {
    const map = {
      zoom: "⌖",
      scope: "◧",
      compare: "⇔",
      recall: "◎",
      scan: "◉",
      trend: "↗",
      treatment: "💉",
      time: "◷",
    };
    return map[icon] || "•";
  }

  function renderTraceTimeline(container, trace, violations) {
    container.innerHTML = "";
    container.className = "trace-timeline";
    (trace || []).forEach(function (line) {
      const item = el("div", "trace-item trace-" + (line.icon || "scope"));
      item.appendChild(el("span", "trace-icon", traceIcon(line.icon)));
      item.appendChild(el("span", "trace-text", line.text || ""));
      container.appendChild(item);
    });
    if (violations && violations.length) {
      const row = el("div", "trace-guard-row");
      violations.forEach(function (v) {
        row.appendChild(el("span", "trace-guard-chip", "claim rejected: not traceable · " + v));
      });
      container.appendChild(row);
    }
  }

  function summarizeScope(scope) {
    if (!scope || typeof scope !== "object") return "";
    const parts = [];
    for (const [k, v] of Object.entries(scope)) {
      if (v == null || v === "") continue;
      parts.push(`${k}=${String(v)}`);
    }
    return parts.join(", ");
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function chip(k, v) {
    const c = el("span", "chip-kv");
    c.appendChild(el("span", "chip-k", k));
    c.appendChild(el("span", "chip-v", String(v)));
    return c;
  }

  function renderRun(question, mode) {
    const run = el("article", "card run");
    run.appendChild(el("p", "qline muted", question || (mode === "deep" ? "Whole-record deep analysis" : "Whole-record investigation")));

    const scope = el("div", "chips run-scope");
    scope.style.display = "none";
    run.appendChild(scope);

    const planHead = el("p", "run-section-title", "Plan");
    planHead.style.display = "none";
    run.appendChild(planHead);
    const plan = el("div", "steps");
    run.appendChild(plan);

    const traceHead = el("p", "run-section-title", mode === "deep" ? "" : "Trace");
    if (mode === "deep") traceHead.style.display = "none";
    run.appendChild(traceHead);
    const steps = el("div", "steps");
    run.appendChild(steps);

    const evidence = el("div", "cards");
    run.appendChild(evidence);

    const status = el("p", "run-status muted small", "thinking…");
    run.appendChild(status);

    root.insertBefore(run, root.firstChild);
    run.scrollIntoView({ behavior: "smooth", block: "nearest" });
    return { run, scope, planHead, plan, traceHead, steps, evidence, status, rows: {}, pending: null, answer: null, buffer: "" };
  }

  function addCoverage(view, summary) {
    view.scope.style.display = "";
    if (typeof summary.glucose_coverage_pct === "number") {
      view.scope.appendChild(chip("coverage", summary.glucose_coverage_pct + "%"));
    }
    view.scope.appendChild(chip("treatment", summary.has_treatment ? "available" : "none"));
    if (summary.limited) {
      const warn = el("p", "flash error",
        "Limited analysis mode: sensor coverage is " + (summary.glucose_coverage_pct || "?") +
        "%. Treat conclusions as provisional.");
      view.run.insertBefore(warn, view.planHead);
    }
  }

  // ── question mode: per-tool trace (the tool shelf) ──
  function addToolCall(view, payload) {
    const line = el("div", "step step-call");
    line.appendChild(el("span", "step-arrow", "→"));
    const scope = summarizeScope(payload.scope || payload.args);
    const label = toolLabel(payload.name);
    line.appendChild(el("span", "step-name", scope ? `${label} (${scope})` : label));
    line.appendChild(el("span", "step-mark step-pending", "…"));
    view.steps.appendChild(line);
    view.pending = line;
  }

  function addToolResult(view, payload) {
    const line = view.pending || view.steps.lastElementChild;
    view.pending = null;
    if (!line) return;
    const mark = line.querySelector(".step-mark");
    if (mark) {
      mark.classList.remove("step-pending");
      mark.classList.add(payload.ok ? "step-ok" : "step-bad");
      mark.textContent = payload.ok ? "✓" : "✗";
    }
  }

  function answerShell(view) {
    if (view.answer) return;
    const card = el("article", "card answer-card");
    card.appendChild(el("p", "run-section-title", "Evidence"));
    view.answer = el("div", "prose answer-body");
    card.appendChild(view.answer);
    view.evidence.appendChild(card);
  }

  function addAnswerDelta(view, payload) {
    answerShell(view);
    view.buffer += payload.delta || "";
    view.answer.innerHTML = escapeHtml(view.buffer).replace(/\n/g, "<br>");
  }

  function addAnswer(view, payload) {
    answerShell(view);
    view.answer.innerHTML = payload.html || escapeHtml(payload.text || "");
    if (payload.trace && payload.trace.length) {
      view.traceHead.style.display = "";
      view.traceHead.textContent = "Trace";
      renderTraceTimeline(view.steps, payload.trace, payload.violations);
    } else if (payload.violations && payload.violations.length) {
      view.traceHead.style.display = "";
      renderTraceTimeline(view.steps, [], payload.violations);
    }
    const chips = el("div", "chips");
    chips.appendChild(chip("faithful", payload.faithful ? "yes" : "flagged"));
    if (payload.tools && payload.tools.length) chips.appendChild(chip("tools", payload.tools.length));
    view.answer.parentElement.appendChild(chips);
    view.answer.parentElement.appendChild(
      el("p", "muted small", "Pattern analysis only. Not a dosing recommendation."),
    );
  }

  // ── deep mode: producer plan + evidence cards ──
  function addPlan(view, steps) {
    view.planHead.style.display = "";
    (steps || []).forEach(function (name) {
      const line = el("div", "step step-call");
      line.appendChild(el("span", "step-arrow", "→"));
      line.appendChild(el("span", "step-name", toolLabel(name)));
      line.appendChild(el("span", "step-mark step-pending", "…"));
      view.plan.appendChild(line);
      view.rows[name] = line;
    });
  }

  function markRunning(view, name) {
    const line = view.rows[name];
    if (line) line.classList.add("step-active");
    view.status.textContent = "running " + toolLabel(name) + "…";
  }

  function markDone(view, name, n) {
    const line = view.rows[name];
    if (!line) return;
    line.classList.remove("step-active");
    const mark = line.querySelector(".step-mark");
    if (mark) {
      mark.classList.remove("step-pending");
      mark.classList.add("step-ok");
      mark.textContent = "✓";
    }
    if (typeof n === "number") line.appendChild(el("span", "step-count muted small", n + " finding(s)"));
  }

  function addNote(view, text) {
    if (!text || /^Planned/.test(text)) return;
    view.traceHead.style.display = "";
    view.traceHead.textContent = "Trace";
    view.steps.appendChild(el("div", "step step-note muted small", text));
  }

  function addEvidenceCard(view, f) {
    const card = el("article", "card finding-card");
    const head = el("div", "card-head");
    head.appendChild(el("h2", null, f.headline || "Finding"));
    if (f.status) head.appendChild(el("span", "badge neutral", f.status));
    card.appendChild(head);
    const chips = el("div", "chips");
    if (f.agent) chips.appendChild(chip("agent", f.agent));
    if (f.scope) chips.appendChild(chip("scope", f.scope));
    if (typeof f.confidence_pct === "number") chips.appendChild(chip("confidence", f.confidence_pct + "%"));
    card.appendChild(chips);
    if (f.body_html) {
      const prose = el("div", "prose");
      prose.innerHTML = f.body_html;
      card.appendChild(prose);
    }
    if (f.skeptic_notes) card.appendChild(el("p", "stats", "Counter-evidence: " + f.skeptic_notes));
    view.evidence.appendChild(card);
  }

  function finish(src, view) {
    src.close();
    if (active === src) active = null;
    input.disabled = false;
    if (deepBtn) deepBtn.disabled = false;
    if (view) view.run.classList.remove("running");
  }

  function start(question, mode) {
    if (active) active.close();
    const view = renderRun(question, mode);
    view.run.classList.add("running");
    input.disabled = true;
    if (deepBtn) deepBtn.disabled = true;

    const url = "/api/investigate/stream?q=" + encodeURIComponent(question) + "&mode=" + mode;
    const src = new EventSource(url);
    active = src;

    src.onmessage = function (e) {
      let event;
      try {
        event = JSON.parse(e.data);
      } catch (_) {
        return;
      }
      const payload = event.payload || {};
      switch (event.kind) {
        case "coverage": addCoverage(view, payload); break;
        case "tool_call": addToolCall(view, payload); break;
        case "tool_result": addToolResult(view, payload); break;
        case "answer_start": answerShell(view); break;
        case "answer_delta": addAnswerDelta(view, payload); break;
        case "answer":
          addAnswer(view, payload);
          view.status.textContent = "done";
          finish(src, view);
          break;
        case "plan": addPlan(view, payload.steps); break;
        case "running": markRunning(view, payload.producer); break;
        case "producer_done": markDone(view, payload.producer, payload.n_findings); break;
        case "step": addNote(view, payload.text); break;
        case "done":
          (payload.findings || []).forEach(function (f) { addEvidenceCard(view, f); });
          view.status.textContent = payload.n_findings > 0
            ? payload.n_findings + " finding(s) · " + (payload.status || "completed")
            : "No durable findings. " + (payload.status || "completed") + ".";
          finish(src, view);
          break;
        case "error":
          view.status.textContent = payload.text || "Investigation failed.";
          view.status.classList.add("error");
          finish(src, view);
          break;
      }
    };

    src.onerror = function () {
      if (active !== src) return;
      if (!view.answer && !view.evidence.childElementCount) {
        view.status.textContent = "Connection lost before the investigation finished.";
        view.status.classList.add("error");
      }
      finish(src, view);
    };
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    start(input.value.trim(), "question");
  });
  if (deepBtn) {
    deepBtn.addEventListener("click", function () {
      start(input.value.trim(), "deep");
    });
  }

  const params = new URLSearchParams(window.location.search);
  const qParam = params.get("q");
  if (qParam) {
    input.value = qParam;
    start(qParam, "question");
    params.delete("q");
    const qs = params.toString();
    window.history.replaceState({}, "", window.location.pathname + (qs ? "?" + qs : ""));
  }
})();
