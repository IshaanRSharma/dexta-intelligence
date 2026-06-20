// Live reasoning stream for the chat page: open an EventSource against the SSE
// endpoint, render each tool call / result as a readable line, and surface the
// final answer prominently. Each ask is independent (the agent has its own
// recall memory), so a follow-up simply opens a fresh stream below the last.

(function () {
  "use strict";

  const form = document.getElementById("ask-form");
  const input = document.getElementById("ask-input");
  const sendBtn = document.getElementById("ask-btn");
  const stopBtn = document.getElementById("stop-btn");
  const root = document.getElementById("stream");
  if (!form || !input || !sendBtn || !root) return;

  const rail = document.getElementById("session-list");
  const newChatBtn = document.getElementById("new-chat-btn");

  const MAX_INPUT_HEIGHT = 200;
  let active = null; // the in-flight EventSource, if any
  let currentView = null; // the run being streamed, if any
  let streaming = false;

  function newSid() {
    return (crypto.randomUUID && crypto.randomUUID()) || String(Date.now()) + Math.random();
  }

  // One conversation id per browser tab so follow-up questions carry context.
  let sid = sessionStorage.getItem("dexta_sid");
  if (!sid) {
    sid = newSid();
    sessionStorage.setItem("dexta_sid", sid);
  }

  function setSid(next) {
    sid = next;
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
    clearEmptyState();
    const run = el("article", "card run");

    const q = el("p", "qline muted", question);
    run.appendChild(q);

    const steps = el("div", "steps");
    run.appendChild(steps);

    const status = el("p", "run-status muted small", "thinking…");
    run.appendChild(status);

    run._answerBody = null;
    run._answerProse = null;
    run._answerBuffer = "";

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

  /** Safe subset renderer - mirrors server ``markdown_to_html`` enough for chat. */
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

  function ensureAnswerShell(view) {
    if (view._answerProse) return;
    if (view.status.isConnected) view.status.remove();
    const answer = el("div", "answer-body answer-streaming");
    const prose = el("div", "prose");
    answer.appendChild(prose);
    view.run.appendChild(answer);
    view._answerBody = answer;
    view._answerProse = prose;
    view._answerBuffer = "";
  }

  function addAnswerStart(view) {
    ensureAnswerShell(view);
  }

  function addAnswerDelta(view, payload) {
    ensureAnswerShell(view);
    view._answerBuffer += payload.delta || "";
    view._answerProse.textContent = view._answerBuffer;
    view.run.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function addAnswer(view, payload) {
    ensureAnswerShell(view);
    setProseHtml(view._answerProse, payload);
    if (view._answerBody) {
      view._answerBody.classList.remove("answer-streaming");
    }
    if (payload.faithful === false) {
      view._answerBody.appendChild(
        el("p", "answer-warn small", "Not all claims could be traced to evidence."),
      );
    }
    if (payload.tools && payload.tools.length) {
      const foot = el("footer", "footnote");
      foot.textContent = "tools: " + payload.tools.join(" · ");
      view._answerBody.appendChild(foot);
    }
    view.run.scrollIntoView({ behavior: "smooth", block: "nearest" });
    view._answerBody = null;
    view._answerProse = null;
    view._answerBuffer = "";
  }

  function addError(view, payload) {
    view.status.remove();
    const err = el("p", "answer-warn", payload.text || "Something went wrong.");
    view.run.appendChild(err);
  }

  function hasText() {
    return input.value.trim().length > 0;
  }

  function resizeInput() {
    input.style.height = "auto";
    const next = Math.min(input.scrollHeight, MAX_INPUT_HEIGHT);
    input.style.height = next + "px";
    input.style.overflowY = input.scrollHeight > MAX_INPUT_HEIGHT ? "auto" : "hidden";
  }

  function updateSendState() {
    sendBtn.disabled = streaming || !hasText();
  }

  function setStreaming(on) {
    streaming = on;
    input.disabled = on;
    form.classList.toggle("is-streaming", on);
    updateSendState();
  }

  function resetInput() {
    input.value = "";
    resizeInput();
    updateSendState();
  }

  function stopAsk() {
    if (active) {
      active.close();
      active = null;
    }
    if (currentView && currentView.status && currentView.status.isConnected) {
      currentView.status.textContent = "stopped";
    }
    currentView = null;
    setStreaming(false);
    input.focus();
  }

  function ask(question) {
    if (active) active.close();
    const view = renderRun(question);
    currentView = view;
    setStreaming(true);

    const src = new EventSource(
      "/api/ask/stream?q=" + encodeURIComponent(question) + "&sid=" + encodeURIComponent(sid),
    );
    active = src;

    function finish() {
      src.close();
      if (active === src) active = null;
      currentView = null;
      setStreaming(false);
      input.focus();
      loadSessions(); // a new/updated conversation may now exist
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
      else if (kind === "answer_start") addAnswerStart(view);
      else if (kind === "answer_delta") addAnswerDelta(view, payload);
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
        if (view.status.isConnected && view.status.textContent === "stopped") {
          finish();
          return;
        }
        addError(view, { text: "Connection lost before an answer arrived." });
      }
      finish();
    };
  }

  // Restore this tab's conversation when returning to the chat page. The server
  // keeps per-session turns in memory; we re-render the completed Q&A pairs so
  // navigating away and back does not lose the thread. (An in-flight ask still
  // ends when you leave the page - only finished turns are restored.)
  function clearEmptyState() {
    const empty = root.querySelector(".chat-empty");
    if (empty) empty.remove();
  }

  function showEmptyState() {
    if (root.querySelector(".run") || root.querySelector(".chat-empty")) return;
    const empty = el("div", "chat-empty");
    empty.appendChild(el("strong", null, "Ask about your glucose, meals, insulin, or patterns"));
    empty.appendChild(
      el("span", "muted small", "Try: “What carbs and boluses did I have today?”"),
    );
    root.appendChild(empty);
  }

  function renderHistoryTurn(question, turn) {
    clearEmptyState();
    const run = el("article", "card run");
    run.appendChild(el("p", "qline muted", question));
    const answer = el("div", "answer-body");
    const prose = el("div", "prose");
    setProseHtml(prose, turn);
    answer.appendChild(prose);
    run.appendChild(answer);
    root.appendChild(run);
  }

  function renderHistoryTurns(turns) {
    root.innerHTML = "";
    if (!turns || !turns.length) {
      showEmptyState();
      return;
    }
    let pendingQ = null;
    for (const t of turns) {
      if (t.role === "user") pendingQ = t.content;
      else if (t.role === "assistant") {
        renderHistoryTurn(pendingQ || "", t);
        pendingQ = null;
      }
    }
  }

  function loadHistory() {
    return fetch("/api/history?sid=" + encodeURIComponent(sid))
      .then(function (r) {
        return r.ok ? r.json() : null;
      })
      .then(function (data) {
        renderHistoryTurns(data && data.turns ? data.turns : []);
        return data;
      })
      .catch(function () {
        showEmptyState();
        return null;
      });
  }

  // ── conversation rail: list past threads, start a new one, switch between ──
  function highlightActive() {
    if (!rail) return;
    for (const item of rail.querySelectorAll(".session-item")) {
      item.classList.toggle("active", item.dataset.sid === sid);
    }
  }

  function switchTo(nextSid) {
    if (nextSid === sid) return;
    stopAsk();
    setSid(nextSid);
    loadHistory();
    highlightActive();
    input.focus();
  }

  function renderSessions(sessions) {
    if (!rail) return;
    rail.innerHTML = "";
    for (const s of sessions) {
      const item = el("div", "session-item");
      item.dataset.sid = s.session_id;

      const openBtn = el("button", "session-open");
      openBtn.type = "button";
      openBtn.appendChild(el("span", "session-preview", s.preview));
      openBtn.appendChild(el("span", "session-meta muted small", s.relative));
      openBtn.addEventListener("click", function () {
        switchTo(s.session_id);
      });

      const delBtn = el("button", "session-delete");
      delBtn.type = "button";
      delBtn.setAttribute("aria-label", "Delete conversation");
      delBtn.innerHTML =
        '<svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/></svg>';
      delBtn.addEventListener("click", function (e) {
        e.preventDefault();
        e.stopPropagation();
        deleteSession(s.session_id);
      });

      item.appendChild(openBtn);
      item.appendChild(delBtn);
      rail.appendChild(item);
    }
    highlightActive();
  }

  function deleteSession(targetSid) {
    fetch("/api/sessions/" + encodeURIComponent(targetSid), { method: "DELETE" })
      .then(function (r) {
        if (!r.ok) return null;
        return r.json();
      })
      .then(function (data) {
        if (!data || !data.ok) return;
        if (targetSid === sid) {
          stopAsk();
          setSid(newSid());
          root.innerHTML = "";
        }
        loadSessions();
        highlightActive();
      })
      .catch(function () {});
  }

  function loadSessions() {
    if (!rail) return Promise.resolve([]);
    return fetch("/api/sessions")
      .then(function (r) {
        return r.ok ? r.json() : null;
      })
      .then(function (data) {
        const sessions = data && data.sessions ? data.sessions : [];
        renderSessions(sessions);
        return sessions;
      })
      .catch(function () {
        return [];
      });
  }

  function bootstrap() {
    loadSessions().then(function (sessions) {
      loadHistory().then(function (data) {
        const turns = data && data.turns ? data.turns : [];
        if (turns.length) {
          highlightActive();
          return;
        }
        if (sessions.length && sessions[0].session_id !== sid) {
          switchTo(sessions[0].session_id);
          return;
        }
        showEmptyState();
        highlightActive();
      });
    });
  }

  if (newChatBtn) {
    newChatBtn.addEventListener("click", function () {
      stopAsk();
      setSid(newSid());
      renderHistoryTurns([]);
      highlightActive();
      input.focus();
    });
  }

  if (stopBtn) {
    stopBtn.addEventListener("click", stopAsk);
  }

  input.addEventListener("input", function () {
    resizeInput();
    updateSendState();
  });

  input.addEventListener("keydown", function (e) {
    if (e.key !== "Enter" || e.shiftKey || e.isComposing) return;
    e.preventDefault();
    if (streaming || !hasText()) return;
    form.requestSubmit();
  });

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    const q = input.value.trim();
    if (!q || streaming) return;
    resetInput();
    ask(q);
  });

  resizeInput();
  updateSendState();
  bootstrap();
})();
