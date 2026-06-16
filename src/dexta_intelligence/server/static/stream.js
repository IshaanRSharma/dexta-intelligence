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

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function inlineMarkdown(text) {
    let out = escapeHtml(text);
    out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
    out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    out = out.replace(/(?<![*\w])\*([^*]+)\*(?![*\w])/g, "<em>$1</em>");
    return out;
  }

  /** Safe subset renderer — mirrors server ``markdown_to_html`` enough for chat. */
  function renderMarkdown(md) {
    const lines = String(md || "").replace(/\r\n/g, "\n").split("\n");
    const out = [];
    let inList = false;
    function closeList() {
      if (inList) {
        out.push("</ul>");
        inList = false;
      }
    }
    for (const raw of lines) {
      const line = raw.trimEnd();
      if (!line.trim()) {
        closeList();
        continue;
      }
      const heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) {
        closeList();
        const level = heading[1].length;
        out.push("<h" + level + ">" + inlineMarkdown(heading[2]) + "</h" + level + ">");
        continue;
      }
      const bullet = line.match(/^\s*-\s+(.*)$/);
      if (bullet) {
        if (!inList) {
          out.push("<ul>");
          inList = true;
        }
        out.push("<li>" + inlineMarkdown(bullet[1]) + "</li>");
        continue;
      }
      closeList();
      out.push("<p>" + inlineMarkdown(line.trim()) + "</p>");
    }
    closeList();
    return out.join("\n");
  }

  function setProseHtml(node, payload) {
    if (payload && payload.html) {
      node.innerHTML = payload.html;
    } else if (payload && payload.text) {
      node.innerHTML = renderMarkdown(payload.text);
    } else {
      node.textContent = "(no answer produced)";
    }
  }

  function addAnswer(view, payload) {
    view.status.remove();
    const answer = el("div", "answer-body");
    const text = el("div", "prose");
    setProseHtml(text, payload);
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

  // Restore this tab's conversation when returning to the chat page. The server
  // keeps per-session turns in memory; we re-render the completed Q&A pairs so
  // navigating away and back does not lose the thread. (An in-flight ask still
  // ends when you leave the page — only finished turns are restored.)
  function renderHistoryTurn(question, turn) {
    const run = el("article", "card run");
    run.appendChild(el("p", "qline muted", question));
    const answer = el("div", "answer-body");
    const prose = el("div", "prose");
    setProseHtml(prose, turn);
    answer.appendChild(prose);
    run.appendChild(answer);
    root.appendChild(run);
  }

  function loadHistory() {
    fetch("/api/history?sid=" + encodeURIComponent(sid))
      .then(function (r) {
        return r.ok ? r.json() : null;
      })
      .then(function (data) {
        if (!data || !data.turns || !data.turns.length) return;
        let pendingQ = null;
        for (const t of data.turns) {
          if (t.role === "user") pendingQ = t.content;
          else if (t.role === "assistant") {
            renderHistoryTurn(pendingQ || "", t);
            pendingQ = null;
          }
        }
        if (root.lastElementChild) {
          root.lastElementChild.scrollIntoView({ block: "nearest" });
        }
      })
      .catch(function () {});
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    ask(q);
  });

  loadHistory();
})();
