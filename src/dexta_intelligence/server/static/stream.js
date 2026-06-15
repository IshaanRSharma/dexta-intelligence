// Live reasoning stream for the chat page: open an EventSource against the SSE
// endpoint, render each tool call / result as a readable line, and surface the
// final answer prominently. Each ask is independent (the agent has its own
// recall memory), so a follow-up simply opens a fresh stream below the last.

(function () {
  "use strict";

  const form = document.getElementById("ask-form");
  const input = document.getElementById("ask-input");
  const button = document.getElementById("ask-btn");
  const root = document.getElementById("stream");
  if (!form || !input || !root) return;

  let active = null; // the in-flight EventSource, if any

  // One conversation id per browser tab so follow-up questions carry context.
  let sid = sessionStorage.getItem("dexta_sid");
  if (!sid) {
    sid = (crypto.randomUUID && crypto.randomUUID()) || String(Date.now()) + Math.random();
    sessionStorage.setItem("dexta_sid", sid);
  }

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  function summarizeArgs(args) {
    if (!args || typeof args !== "object") return "";
    const parts = [];
    for (const [k, v] of Object.entries(args)) {
      if (v == null || v === "") continue;
      parts.push(`${k}=${String(v)}`);
    }
    return parts.join(", ");
  }

  function toolLabel(name) {
    return String(name || "tool").replace(/_/g, " ");
  }

  function renderRun(question) {
    const run = el("article", "card run");

    const q = el("p", "qline muted", question);
    run.appendChild(q);

    const steps = el("div", "steps");
    run.appendChild(steps);

    const status = el("p", "run-status muted small", "thinking…");
    run.appendChild(status);

    root.appendChild(run);
    run.scrollIntoView({ behavior: "smooth", block: "nearest" });
    return { run, steps, status };
  }

  function addToolCall(view, payload) {
    const line = el("div", "step step-call");
    line.appendChild(el("span", "step-arrow", "→"));
    const label = toolLabel(payload.name);
    const args = summarizeArgs(payload.args);
    line.appendChild(el("span", "step-name", args ? `${label} (${args})` : label));
    line.appendChild(el("span", "step-mark step-pending", "…"));
    view.steps.appendChild(line);
    view._pending = line; // matched by the next tool_result
  }

  function addToolResult(view, payload) {
    const line = view._pending || view.steps.lastElementChild;
    view._pending = null;
    if (!line) return;
    const mark = line.querySelector(".step-mark");
    if (mark) {
      mark.classList.remove("step-pending");
      mark.classList.add(payload.ok ? "step-ok" : "step-bad");
      mark.textContent = payload.ok ? "✓" : "✗";
    }
  }

  function addAnswer(view, payload) {
    view.status.remove();
    const answer = el("div", "answer-body");
    const text = el("div", "prose");
    text.textContent = payload.text || "(no answer produced)";
    answer.appendChild(text);
    if (payload.faithful === false) {
      answer.appendChild(el("p", "answer-warn small", "Not all claims could be traced to evidence."));
    }
    if (payload.tools && payload.tools.length) {
      const foot = el("footer", "footnote");
      foot.textContent = "tools: " + payload.tools.join(" · ");
      answer.appendChild(foot);
    }
    view.run.appendChild(answer);
    view.run.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function addError(view, payload) {
    view.status.remove();
    const err = el("p", "answer-warn", payload.text || "Something went wrong.");
    view.run.appendChild(err);
  }

  function setBusy(busy) {
    input.disabled = busy;
    button.disabled = busy;
    button.textContent = busy ? "…" : "Ask";
  }

  function ask(question) {
    if (active) active.close();
    const view = renderRun(question);
    setBusy(true);

    const src = new EventSource(
      "/api/ask/stream?q=" + encodeURIComponent(question) + "&sid=" + encodeURIComponent(sid),
    );
    active = src;

    function finish() {
      src.close();
      if (active === src) active = null;
      setBusy(false);
      input.focus();
    }

    src.onmessage = function (e) {
      let event;
      try {
        event = JSON.parse(e.data);
      } catch (_) {
        return;
      }
      const kind = event.kind;
      const payload = event.payload || {};
      if (kind === "tool_call") addToolCall(view, payload);
      else if (kind === "tool_result") addToolResult(view, payload);
      else if (kind === "answer") {
        addAnswer(view, payload);
        finish();
      } else if (kind === "error") {
        addError(view, payload);
        finish();
      }
    };

    src.onerror = function () {
      // EventSource fires onerror on normal stream close too; only surface it
      // if we never reached a terminal event.
      if (active !== src) return;
      if (!view.run.querySelector(".answer-body") && !view.run.querySelector(".answer-warn")) {
        addError(view, { text: "Connection lost before an answer arrived." });
      }
      finish();
    };
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    ask(q);
  });
})();
