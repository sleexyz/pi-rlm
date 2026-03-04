// ── State ───────────────────────────────────────────────────────────

let sessions = [];
let resultMap = {}; // taskId -> { correct, cost, tokens, model }
let activeSessionId = null;
let currentTurnCount = 0;
let activeSessions = new Set();
let streamingEl = null; // live-updating element for message_update deltas
let streamingText = "";
let streamingThinking = "";

// Dashboard state
let dashboardMode = false;
/** @type {Map<string, { streamingEl: Element|null, streamingText: string, streamingThinking: string, turnCount: number }>} */
const panelStates = new Map();

const sidebar = document.getElementById("sidebar");
const trace = document.getElementById("trace");
const dashboard = document.getElementById("dashboard");
const btnDashboard = document.getElementById("btn-dashboard");

// ── Load data ───────────────────────────────────────────────────────

async function init() {
  const [sessData, resData] = await Promise.all([
    fetch("/api/sessions").then((r) => r.json()),
    fetch("/api/results").then((r) => r.json()),
  ]);

  sessions = sessData;

  // Build result map from all result files
  for (const rf of resData) {
    const model = rf.config?.model || "";
    for (const r of rf.results) {
      resultMap[r.taskId] = { correct: r.correct, failed: r.failed, cost: r.cost, tokens: r.tokens, model };
    }
  }

  // Populate activeSessions from session data
  for (const s of sessions) {
    if (s.active) activeSessions.add(s.sessionId);
  }

  renderSidebar();
  connectWS();
}

// ── Sidebar ─────────────────────────────────────────────────────────

function renderSidebar() {
  if (sessions.length === 0) {
    sidebar.innerHTML =
      '<div class="no-sessions">No sessions found.<br>Run arc-runner with --log to create logs.</div>';
    return;
  }

  // Group sessions by domain
  const grouped = {};
  for (const s of sessions) {
    const d = s.domain || "unknown";
    if (!grouped[d]) grouped[d] = [];
    grouped[d].push(s);
  }

  const domains = Object.keys(grouped).sort();
  const multiDomain = domains.length > 1;

  let html = "";
  for (const domain of domains) {
    if (multiDomain) {
      html += '<div class="domain-header">' + esc(domain) + "</div>";
    }
    for (const s of grouped[domain]) {
      const label = s.taskId || s.sessionId;
      const result = s.taskId ? resultMap[s.taskId] : null;
      const isLive = activeSessions.has(s.sessionId);
      const badge = isLive
        ? '<span class="badge live">LIVE</span>'
        : result
          ? result.correct
            ? '<span class="badge ok">CORRECT</span>'
            : result.failed
              ? '<span class="badge fail">FAILED</span>'
              : '<span class="badge err">WRONG</span>'
          : "";
      const domainTag = !multiDomain && s.domain
        ? '<span class="domain-tag">' + esc(s.domain) + "</span>"
        : "";
      const cost = s.usage ? "$" + s.usage.totalCost.toFixed(4) : "";
      const tokens = s.usage ? s.usage.totalTokens.toLocaleString() + " tok" : "";
      const model = s.model || (result && result.model) || "";
      const modelShort = model.replace("claude-", "").replace(/-\d{8,}$/, "");
      const date = s.ts ? friendlyDate(s.ts) : "";
      const line1 = [s.turns + " turns", cost, tokens].filter(Boolean).join(" \u00b7 ");
      const line2 = [modelShort, date].filter(Boolean).join(" \u00b7 ");

      html +=
        '<div class="session-item" data-id="' +
        esc(s.sessionId) +
        '">' +
        '<span class="task-id">' +
        esc(label) +
        "</span>" +
        domainTag +
        badge +
        '<div class="meta">' +
        esc(line1) +
        "</div>" +
        (line2 ? '<div class="meta">' + esc(line2) + "</div>" : "") +
        "</div>";
    }
  }
  sidebar.innerHTML = html;

  sidebar.querySelectorAll(".session-item").forEach((el) => {
    el.addEventListener("click", () => {
      if (dashboardMode) exitDashboard();
      loadTrace(el.dataset.id);
    });
  });

  // Re-apply active highlight if a session is selected
  if (activeSessionId) {
    sidebar.querySelectorAll(".session-item").forEach((el) => {
      el.classList.toggle("active", el.dataset.id === activeSessionId);
    });
  }
}

async function refreshSessions() {
  const sessData = await fetch("/api/sessions").then((r) => r.json());
  sessions = sessData;
  renderSidebar();
}

// ── Trace loading ───────────────────────────────────────────────────

async function loadTrace(sessionId) {
  if (dashboardMode) exitDashboard();

  // Unsubscribe old, subscribe new
  if (ws && activeSessionId && activeSessionId !== sessionId) {
    wsSend({ type: "unsubscribe", sessionId: activeSessionId });
  }

  activeSessionId = sessionId;

  sidebar.querySelectorAll(".session-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.id === sessionId);
  });

  // If session is active, use subscription-based streaming
  if (activeSessions.has(sessionId) && ws) {
    // Clear trace and let replay fill it
    trace.className = "";
    trace.innerHTML = "";
    currentTurnCount = 0;
    streamingEl = null;
    streamingText = "";
    streamingThinking = "";
    wsSend({ type: "subscribe", sessionId });
  } else {
    // Completed session — load via HTTP
    const events = await fetch("/api/session/" + encodeURIComponent(sessionId)).then((r) =>
      r.json(),
    );
    renderTrace(events);
  }
}

// ── Trace rendering ─────────────────────────────────────────────────

function renderSingleEvent(evt, container) {
  if (!container) container = trace;
  const depth = evt.depth ?? 0;
  const depthClass = "depth-" + Math.min(depth, 3);

  // Get the streaming state for this container
  const state = getContainerState(container);

  switch (evt.type) {
    case "session_start": {
      const div = mk("div", "event verbose " + depthClass);
      div.innerHTML = '<span class="dim">Session: ' + esc(evt.sessionId || "") + "</span>";
      container.appendChild(div);
      break;
    }

    case "agent_start": {
      const label = evt.label || "Agent";
      const idTag = evt.agentId != null ? "#" + evt.agentId + " " : "";

      const div = mk("div", "event verbose " + depthClass);
      const ts = evt.ts ? " " + timeStr(evt.ts) : "";
      div.innerHTML =
        '<div class="agent-header">\u250c\u2500 ' +
        esc(idTag + label) +
        " " +
        "\u2500".repeat(Math.max(0, 50 - (idTag + label).length)) +
        '<span class="ts">' + esc(ts) + '</span>' +
        "</div>";

      if (evt.systemPrompt) {
        const details = document.createElement("details");
        details.className = "sys-block";
        details.innerHTML =
          '<summary class="sys">SYS  System Prompt</summary>' +
          '<div class="collapsible-content sys">' +
          esc(evt.systemPrompt) +
          "</div>";
        div.appendChild(details);
      }

      if (evt.userMessage) {
        const um = mk("pre", "usr");
        um.textContent = "USR  " + evt.userMessage;
        div.appendChild(um);
      }

      container.appendChild(div);
      break;
    }

    case "turn_start": {
      state.turnCount++;
      if (state.turnCount > 1) {
        const hr = document.createElement("hr");
        hr.className = "turn-sep verbose";
        hr.dataset.turn = "Turn " + state.turnCount + (evt.ts ? "  " + timeStr(evt.ts) : "");
        container.appendChild(hr);
      }
      break;
    }

    case "message_update": {
      if (!evt.delta) break;
      if (evt.deltaType === "thinking_delta") {
        state.streamingThinking += evt.delta;
        if (!state.streamingEl) {
          state.streamingEl = mk("div", "event " + depthClass);
          const pre = mk("pre", "thk streaming-text");
          state.streamingEl.appendChild(pre);
          container.appendChild(state.streamingEl);
        }
        const pre = state.streamingEl.querySelector(".streaming-text");
        if (pre) pre.textContent = state.streamingThinking;
      } else if (evt.deltaType === "text_delta") {
        state.streamingText += evt.delta;
        if (state.streamingThinking && state.streamingEl) {
          const thinkPre = state.streamingEl.querySelector(".streaming-text");
          if (thinkPre) {
            const details = document.createElement("details");
            details.className = "event thk-block " + depthClass;
            const preview = state.streamingThinking.split("\n")[0].slice(0, 80);
            details.innerHTML =
              '<summary class="thk">THK  ' + esc(preview) + '\u2026</summary>' +
              '<div class="collapsible-content thk">' + esc(state.streamingThinking) + '</div>';
            state.streamingEl.replaceWith(details);
          }
          state.streamingThinking = "";
          state.streamingEl = null;
        }
        if (!state.streamingEl) {
          state.streamingEl = mk("div", "event llm " + depthClass);
          const pre = mk("pre", "llm streaming-text");
          state.streamingEl.appendChild(pre);
          container.appendChild(state.streamingEl);
        }
        const pre = state.streamingEl.querySelector(".streaming-text");
        if (pre) pre.textContent = state.streamingText;
      }
      break;
    }

    case "message_end": {
      state.streamingText = "";
      state.streamingThinking = "";
      if (state.streamingEl) {
        state.streamingEl.remove();
        state.streamingEl = null;
      }
      if (!evt.content) break;

      for (const block of evt.content) {
        if (block.type === "thinking" && block.thinking) {
          const details = document.createElement("details");
          details.className = "event thk-block " + depthClass;
          const preview = block.thinking.split("\n")[0].slice(0, 80);
          details.innerHTML =
            '<summary class="thk">THK  ' +
            esc(preview) +
            "\u2026</summary>" +
            '<div class="collapsible-content thk">' +
            esc(block.thinking) +
            "</div>";
          container.appendChild(details);
        }

        if (block.type === "text" && block.text) {
          const div = mk("div", "event llm " + depthClass);
          const pre = mk("pre", "llm");
          pre.textContent = block.text;
          div.appendChild(pre);
          container.appendChild(div);
        }

        if (block.type === "tool_use" && block.name) {
          const div = mk("div", "event eval " + depthClass);
          const code = block.arguments?.code || block.input?.code || "";
          if (code) {
            const details = document.createElement("details");
            details.open = true;
            details.innerHTML =
              '<summary class="eval">TOOL ' +
              esc(block.name) +
              "</summary>" +
              '<div class="collapsible-content eval">' +
              esc(code) +
              "</div>";
            div.appendChild(details);
          } else {
            div.innerHTML = '<span class="eval">TOOL ' + esc(block.name) + "</span>";
          }
          container.appendChild(div);
        }
      }
      break;
    }

    case "tool_execution_start": {
      const code = evt.code;
      if (code) {
        const details = document.createElement("details");
        details.className = "event " + depthClass;
        details.open = true;
        details.innerHTML =
          '<summary class="eval">EVAL ' +
          esc(evt.toolName || "code") +
          "</summary>" +
          '<div class="collapsible-content eval">' +
          esc(code) +
          "</div>";
        container.appendChild(details);
      } else if (evt.toolName) {
        const div = mk("div", "event verbose " + depthClass);
        div.innerHTML = '<span class="dim">' + esc(evt.toolName) + "\u2026</span>";
        container.appendChild(div);
      }
      break;
    }

    case "tool_execution_end": {
      const resultText = formatResult(evt.result);
      if (resultText) {
        const details = document.createElement("details");
        details.className = "event " + depthClass;
        const preview = resultText.split("\n")[0].slice(0, 80);
        const hasError = resultText.includes("ERROR:");
        const hasOk = /^→|accuracy|correct|score/im.test(resultText);
        const cls = hasError ? "err" : hasOk ? "ok" : "output";

        details.innerHTML =
          '<summary class="' +
          cls +
          '">OUT  ' +
          esc(preview) +
          "</summary>" +
          '<div class="collapsible-content">' +
          colorizeOutput(resultText) +
          "</div>";
        container.appendChild(details);
      }
      break;
    }

    case "agent_end": {
      const div = mk("div", "event verbose " + depthClass);
      div.innerHTML =
        '<div class="agent-footer">\u2514' + "\u2500".repeat(55) + "</div>";
      container.appendChild(div);
      break;
    }

    case "session_end": {
      const u = evt.usage;
      if (u) {
        const div = mk("div", "usage-summary");
        const tokens = (u.totalTokens || 0).toLocaleString();
        const cost = "$" + (u.totalCost || 0).toFixed(4);
        div.innerHTML =
          "\u2500\u2500 Usage \u2500\u2500<br>Tokens: " +
          tokens +
          " \u00b7 Cost: " +
          cost +
          " \u00b7 Turns: " +
          state.turnCount;
        container.appendChild(div);
      }
      break;
    }
  }
}

/** Get or create per-container streaming state. For trace view, uses globals. */
function getContainerState(container) {
  if (container === trace) {
    return {
      get turnCount() { return currentTurnCount; },
      set turnCount(v) { currentTurnCount = v; },
      get streamingEl() { return streamingEl; },
      set streamingEl(v) { streamingEl = v; },
      get streamingText() { return streamingText; },
      set streamingText(v) { streamingText = v; },
      get streamingThinking() { return streamingThinking; },
      set streamingThinking(v) { streamingThinking = v; },
    };
  }
  // Dashboard panel — use panelStates keyed by panel's sessionId
  const sid = container.dataset?.sessionId;
  if (sid && panelStates.has(sid)) {
    return panelStates.get(sid);
  }
  // Fallback: create ephemeral state
  return { turnCount: 0, streamingEl: null, streamingText: "", streamingThinking: "" };
}

function renderTrace(events) {
  trace.className = "";
  trace.innerHTML = "";
  currentTurnCount = 0;
  streamingEl = null;
  streamingText = "";
  streamingThinking = "";

  const turnFilter = parseInt(document.getElementById("f-turn").value, 10);
  let turnNum = 0;
  let inFilteredTurn = true;

  for (const evt of events) {
    if (evt.type === "turn_start") {
      turnNum++;
      inFilteredTurn = turnFilter < 0 || turnNum === turnFilter;
      if (!inFilteredTurn) continue;
    }
    if (!inFilteredTurn && (evt.type === "message_end" || evt.type === "tool_execution_start" || evt.type === "tool_execution_end")) continue;

    renderSingleEvent(evt, trace);
  }

  if (trace.children.length === 0) {
    trace.className = "empty";
    trace.textContent = "No events to display";
  }

  trace.scrollTop = 0;
  applyFilters();
}

// ── Live event streaming ────────────────────────────────────────────

function appendLiveEvent(evt, container) {
  if (!container) container = trace;

  // Remove "empty" state if this is the first live event
  if (container === trace && trace.className === "empty") {
    trace.className = "";
    trace.innerHTML = "";
  }

  // Auto-scroll if user is near the bottom (within 150px)
  const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 150;

  renderSingleEvent(evt, container);
  if (container === trace) applyFilters();

  if (nearBottom) {
    container.scrollTop = container.scrollHeight;
  }
}

// ── Dashboard mode ──────────────────────────────────────────────────

function enterDashboard() {
  dashboardMode = true;
  btnDashboard.classList.add("active");
  trace.classList.add("hidden");
  sidebar.classList.add("hidden");
  dashboard.classList.remove("hidden");
  dashboard.innerHTML = "";
  panelStates.clear();

  // Create panels for all active sessions
  for (const sid of activeSessions) {
    createDashboardPanel(sid);
  }

  if (dashboard.children.length === 0) {
    dashboard.innerHTML = '<div class="no-sessions">No active sessions. Start a run with --log --stream.</div>';
  }

  // Subscribe to all sessions
  wsSend({ type: "subscribe_all" });
}

function exitDashboard() {
  dashboardMode = false;
  btnDashboard.classList.remove("active");
  dashboard.classList.add("hidden");
  dashboard.innerHTML = "";
  sidebar.classList.remove("hidden");
  trace.classList.remove("hidden");
  panelStates.clear();

  // Unsubscribe all, re-subscribe to active session if any
  wsSend({ type: "unsubscribe_all" });
  if (activeSessionId) {
    wsSend({ type: "subscribe", sessionId: activeSessionId });
  }
}

function toggleDashboard() {
  if (dashboardMode) {
    exitDashboard();
  } else {
    enterDashboard();
  }
}

function createDashboardPanel(sessionId) {
  if (dashboard.querySelector(`[data-session-id="${sessionId}"]`)) return;

  // Find session info for label
  const sessionInfo = sessions.find((s) => s.sessionId === sessionId);
  const label = sessionInfo?.taskId || sessionId;

  const panel = mk("div", "dashboard-panel");
  panel.dataset.sessionId = sessionId;
  panel.innerHTML =
    '<div class="panel-header">' +
    '<span>' + esc(label) + '</span>' +
    '<span class="panel-status">LIVE</span>' +
    '</div>' +
    '<div class="panel-body" data-session-id="' + esc(sessionId) + '"></div>';

  // Click header to switch to single-session view
  panel.querySelector(".panel-header").addEventListener("click", () => {
    exitDashboard();
    loadTrace(sessionId);
  });

  dashboard.appendChild(panel);

  // Create panel state
  panelStates.set(sessionId, {
    turnCount: 0,
    streamingEl: null,
    streamingText: "",
    streamingThinking: "",
  });
}

function getDashboardBody(sessionId) {
  return dashboard.querySelector(`.panel-body[data-session-id="${sessionId}"]`);
}

btnDashboard.addEventListener("click", toggleDashboard);

// ── WebSocket client ────────────────────────────────────────────────

let ws = null;
let replayingSession = null; // sessionId currently being replayed

function wsSend(msg) {
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify(msg));
  }
}

function connectWS() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(proto + "//" + location.host + "/ws");

  ws.onopen = () => {
    // Re-subscribe after reconnect
    if (dashboardMode) {
      wsSend({ type: "subscribe_all" });
    } else if (activeSessionId) {
      wsSend({ type: "subscribe", sessionId: activeSessionId });
    }
  };

  ws.onmessage = (e) => {
    let msg;
    try {
      msg = JSON.parse(e.data);
    } catch {
      return;
    }

    switch (msg.type) {
      case "active_sessions": {
        for (const id of msg.sessionIds) {
          activeSessions.add(id);
        }
        renderSidebar();
        break;
      }

      case "session_active": {
        activeSessions.add(msg.sessionId);
        refreshSessions();
        if (dashboardMode) {
          createDashboardPanel(msg.sessionId);
          // Remove "no sessions" placeholder
          const placeholder = dashboard.querySelector(".no-sessions");
          if (placeholder) placeholder.remove();
          wsSend({ type: "subscribe", sessionId: msg.sessionId });
        }
        break;
      }

      case "replay_start": {
        replayingSession = msg.sessionId;
        break;
      }

      case "replay_end": {
        replayingSession = null;
        break;
      }

      case "replay_complete": {
        replayingSession = null;
        break;
      }

      case "event": {
        if (msg.event?.ts) updateLastEvent(msg.event.ts);

        if (dashboardMode) {
          // Route to correct dashboard panel
          const body = getDashboardBody(msg.sessionId);
          if (body) {
            appendLiveEvent(msg.event, body);
          }
        } else if (msg.sessionId === activeSessionId) {
          appendLiveEvent(msg.event, trace);
        }
        break;
      }

      case "session_ended": {
        activeSessions.delete(msg.sessionId);
        renderSidebar();

        if (dashboardMode) {
          // Mark panel as ended
          const panel = dashboard.querySelector(`.dashboard-panel[data-session-id="${msg.sessionId}"]`);
          if (panel) {
            const status = panel.querySelector(".panel-status");
            if (status) {
              status.textContent = "ENDED";
              status.style.animation = "none";
              status.style.color = "var(--fg-dim)";
            }
          }
        } else if (msg.sessionId === activeSessionId) {
          // Reload full trace for clean final render
          loadTrace(activeSessionId);
        }
        break;
      }
    }
  };

  ws.onclose = () => {
    ws = null;
    setTimeout(connectWS, 2000);
  };

  ws.onerror = () => {
    ws?.close();
  };
}

// ── Helpers ─────────────────────────────────────────────────────────

function mk(tag, cls) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  return e;
}

function esc(s) {
  if (!s) return "";
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function timeStr(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function friendlyDate(ts) {
  const d = new Date(ts * 1000);
  const now = new Date();
  const diff = now - d;
  const mins = Math.floor(diff / 60000);
  const hrs = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);
  if (mins < 1) return "just now";
  if (mins < 60) return mins + "m ago";
  if (hrs < 24) return hrs + "h ago";
  if (days < 7) return days + "d ago";
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function formatResult(result) {
  if (!result) return "";
  const content = result.content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((c) => c.type === "text" && c.text)
    .map((c) => c.text.trim())
    .join("\n")
    .trim();
}

function colorizeOutput(text) {
  return text
    .split("\n")
    .map((line) => {
      if (line.startsWith("ERROR:")) return '<span class="err">' + esc(line) + "</span>";
      if (/^→|accuracy|correct|score/i.test(line))
        return '<span class="ok">' + esc(line) + "</span>";
      return '<span class="output">' + esc(line) + "</span>";
    })
    .join("\n");
}

// ── Filters ─────────────────────────────────────────────────────────

function applyFilters() {
  document.body.classList.toggle("hide-thinking", document.getElementById("f-thinking").checked);
  document.body.classList.toggle("hide-sysprompt", document.getElementById("f-sysprompt").checked);
  document.body.classList.toggle("compact", document.getElementById("f-compact").checked);
}

document.getElementById("f-thinking").addEventListener("change", applyFilters);
document.getElementById("f-sysprompt").addEventListener("change", applyFilters);
document.getElementById("f-compact").addEventListener("change", applyFilters);
document.getElementById("f-turn").addEventListener("change", () => {
  if (activeSessionId) loadTrace(activeSessionId);
});

// ── Keyboard shortcuts ──────────────────────────────────────────────

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

  switch (e.key) {
    case "j": {
      const seps = trace.querySelectorAll(".turn-sep");
      const scrollTop = trace.scrollTop;
      for (const sep of seps) {
        if (sep.offsetTop > scrollTop + 10) {
          trace.scrollTo({ top: sep.offsetTop - 8, behavior: "smooth" });
          break;
        }
      }
      break;
    }
    case "k": {
      const seps = [...trace.querySelectorAll(".turn-sep")].reverse();
      const scrollTop = trace.scrollTop;
      for (const sep of seps) {
        if (sep.offsetTop < scrollTop - 10) {
          trace.scrollTo({ top: sep.offsetTop - 8, behavior: "smooth" });
          break;
        }
      }
      break;
    }
    case "e": {
      trace.querySelectorAll("details").forEach((d) => (d.open = true));
      break;
    }
    case "c": {
      trace.querySelectorAll("details").forEach((d) => (d.open = false));
      break;
    }
    case "t": {
      const cb = document.getElementById("f-thinking");
      cb.checked = !cb.checked;
      applyFilters();
      break;
    }
    case "s": {
      const cb = document.getElementById("f-sysprompt");
      cb.checked = !cb.checked;
      applyFilters();
      break;
    }
    case "d": {
      toggleDashboard();
      break;
    }
  }
});

// ── Last event indicator ─────────────────────────────────────────────

let lastEventTs = 0;
const lastEventEl = document.getElementById("last-event");

function updateLastEvent(ts) {
  if (ts && ts > lastEventTs) lastEventTs = ts;
}

setInterval(() => {
  if (!lastEventTs || !activeSessions.has(activeSessionId)) {
    lastEventEl.textContent = "";
    return;
  }
  const ago = Math.floor((Date.now() / 1000) - lastEventTs);
  if (ago < 5) lastEventEl.textContent = "streaming...";
  else if (ago < 60) lastEventEl.textContent = ago + "s since last event";
  else lastEventEl.textContent = Math.floor(ago / 60) + "m " + (ago % 60) + "s since last event";
  lastEventEl.style.color = ago > 120 ? "var(--red)" : ago > 30 ? "var(--yellow)" : "var(--green)";
}, 1000);

// ── Init ────────────────────────────────────────────────────────────

init();
