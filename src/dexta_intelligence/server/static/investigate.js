// Live investigation stream for the Investigations page: open an EventSource
// against /api/investigate/stream, render the coordinator's plan, narrate each
// producer as it runs, and surface the resulting evidence cards. Pure read-only
// rendering; the server persists the run and findings.

(function () {
  "use strict";

  const form = document.getElementById("investigate-form");
  const input = document.getElementById("investigate-q");
  const root = document.getElementById("investigate-stream");
  if (!form || !input || !root) return;

  let active = null; // in-flight EventSource

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  function producerLabel(name) {
    return String(name || "producer").replace(/_/g, " ");
  }

  function renderRun(question) {
    const run = el("article", "card run");
    run.appendChild(el("p", "qline muted", question || "Whole-record investigation"));

    const scope = el("div", "chips run-scope");
    scope.style.display = "none";
    run.appendChild(scope);

    const planHead = el("p", "run-section-title", "Plan");
    planHead.style.display = "none";
    run.appendChild(planHead);
    const plan = el("div", "steps");
    run.appendChild(plan);

    const notes = el("div", "steps run-notes");
    run.appendChild(notes);

    const evidence = el("div", "cards");
    run.appendChild(evidence);

    const status = el("p", "run-status muted small", "planning…");
    run.appendChild(status);

    root.insertBefore(run, root.firstChild);
    run.scrollIntoView({ behavior: "smooth", block: "nearest" });
    return { run, scope, planHead, plan, notes, evidence, status, rows: {} };
  }

  function addCoverage(view, summary) {
    view.scope.style.display = "";
    if (typeof summary.glucose_coverage_pct === "number") {
      view.scope.appendChild(chip("coverage", summary.glucose_coverage_pct + "%"));
    }
    view.scope.appendChild(chip("treatment", summary.has_treatment ? "available" : "none"));
    if (summary.limited) {
      const warn = el(
        "p",
        "flash error",
        "Limited analysis mode: sensor coverage is " +
          (summary.glucose_coverage_pct || "?") +
          "%. Treat conclusions as provisional.",
      );
      view.run.insertBefore(warn, view.planHead);
    }
  }

  function addPlan(view, steps) {
    view.planHead.style.display = "";
    (steps || []).forEach(function (name) {
      const line = el("div", "step step-call");
      line.appendChild(el("span", "step-arrow", "→"));
      line.appendChild(el("span", "step-name", producerLabel(name)));
      line.appendChild(el("span", "step-mark step-pending", "…"));
      view.plan.appendChild(line);
      view.rows[name] = line;
    });
  }

  function markRunning(view, name) {
    const line = view.rows[name];
    if (line) line.classList.add("step-active");
    view.status.textContent = "running " + producerLabel(name) + "…";
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
    if (typeof n === "number") {
      line.appendChild(el("span", "step-count muted small", n + " finding(s)"));
    }
  }

  function addNote(view, text) {
    if (!text || /^Planned/.test(text)) return; // plan already rendered above
    view.notes.appendChild(el("div", "step step-note muted small", text));
  }

  function chip(k, v) {
    const c = el("span", "chip-kv");
    c.appendChild(el("span", "chip-k", k));
    c.appendChild(el("span", "chip-v", String(v)));
    return c;
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
    if (f.skeptic_notes) {
      card.appendChild(el("p", "stats", "Counter-evidence: " + f.skeptic_notes));
    }
    view.evidence.appendChild(card);
  }

  function finish(src, view) {
    src.close();
    if (active === src) active = null;
    input.disabled = false;
    if (view) view.run.classList.remove("running");
  }

  function start(question) {
    if (active) active.close();
    const view = renderRun(question);
    view.run.classList.add("running");
    input.disabled = true;

    const src = new EventSource("/api/investigate/stream?q=" + encodeURIComponent(question));
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
        case "coverage":
          addCoverage(view, payload);
          break;
        case "plan":
          addPlan(view, payload.steps);
          break;
        case "running":
          markRunning(view, payload.producer);
          break;
        case "producer_done":
          markDone(view, payload.producer, payload.n_findings);
          break;
        case "step":
          addNote(view, payload.text);
          break;
        case "done":
          (payload.findings || []).forEach(function (f) {
            addEvidenceCard(view, f);
          });
          view.status.textContent =
            payload.n_findings > 0
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
      if (!view.evidence.childElementCount) {
        view.status.textContent = "Connection lost before the investigation finished.";
        view.status.classList.add("error");
      }
      finish(src, view);
    };
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    const question = input.value.trim();
    start(question);
  });
})();
