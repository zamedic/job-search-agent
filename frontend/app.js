// JobScope — frontend
// No build, no framework. Just fetch + EventSource-style streaming.

const $ = (id) => document.getElementById(id);

const state = {
  sessions: [],
  currentSessionId: null,
  inFlight: null,   // current AbortController for a streaming turn
  currentActivity: null, // div where we render search chips during a turn
};

// ----------------------------------------------------------------------
// Markdown renderer (small, no dependencies).
// Handles: headings, bold, italic, inline code, fenced code blocks,
// unordered/ordered lists, links, blockquotes, tables, horizontal rules,
// and paragraphs. Good enough for an LLM-emitted report.
// ----------------------------------------------------------------------
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function renderInline(text) {
  // Escape first, then apply inline patterns.
  let s = escapeHtml(text);
  // links: [text](url)
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    (_, t, u) => `<a href="${u}" target="_blank" rel="noopener noreferrer">${t}</a>`);
  // inline code
  s = s.replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`);
  // bold
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/__([^_]+)__/g, "<strong>$1</strong>");
  // italic
  s = s.replace(/(^|\W)\*([^*\s][^*]*)\*(?=\W|$)/g, "$1<em>$2</em>");
  return s;
}

function renderMarkdown(md) {
  if (!md) return "";
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  const out = [];
  let i = 0;

  const isTableDivider = (l) => new RegExp("^\\s*\\|?[\\s:|-]+\\|[\\s:|-]+\\s*\\|?\\s*$").test(l);

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block
    if (/^```/.test(line)) {
      const lang = line.replace(/^```/, "").trim();
      const codeLines = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i])) {
        codeLines.push(lines[i]);
        i++;
      }
      i++; // skip closing fence
      const langClass = lang ? ` class="language-${escapeHtml(lang)}"` : "";
      out.push(`<pre><code${langClass}>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
      continue;
    }

    // Headings
    const h = /^(#{1,3})\s+(.*)$/.exec(line);
    if (h) {
      out.push(`<h${h[1].length}>${renderInline(h[2])}</h${h[1].length}>`);
      i++;
      continue;
    }

    // Horizontal rule
    if (/^\s*---+\s*$/.test(line)) {
      out.push("<hr>");
      i++;
      continue;
    }

    // Tables: need a header row, a divider, then body rows.
    if (line.trim().startsWith("|") && i + 1 < lines.length && isTableDivider(lines[i + 1])) {
      const headerCells = line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      i += 2; // skip header + divider
      const bodyRows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        bodyRows.push(lines[i].trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
        i++;
      }
      out.push("<table><thead><tr>" +
        headerCells.map((c) => `<th>${renderInline(c)}</th>`).join("") +
        "</tr></thead><tbody>" +
        bodyRows.map((row) => "<tr>" + row.map((c) => `<td>${renderInline(c)}</td>`).join("") + "</tr>").join("") +
        "</tbody></table>");
      continue;
    }

    // Blockquote
    if (/^>\s?/.test(line)) {
      const quoteLines = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) {
        quoteLines.push(lines[i].replace(/^>\s?/, ""));
        i++;
      }
      out.push(`<blockquote>${renderInline(quoteLines.join(" "))}</blockquote>`);
      continue;
    }

    // Unordered list
    if (/^[-*]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^[-*]\s+/, ""));
        i++;
      }
      out.push("<ul>" + items.map((it) => `<li>${renderInline(it)}</li>`).join("") + "</ul>");
      continue;
    }

    // Ordered list
    if (/^\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\d+\.\s+/, ""));
        i++;
      }
      out.push("<ol>" + items.map((it) => `<li>${renderInline(it)}</li>`).join("") + "</ol>");
      continue;
    }

    // Blank line
    if (line.trim() === "") {
      i++;
      continue;
    }

    // Paragraph: collect consecutive non-blank, non-block lines.
    const para = [];
    while (i < lines.length
           && lines[i].trim() !== ""
           && !/^(#{1,3}\s|```|>\s?|[-*]\s|\d+\.\s|---+\s*$)/.test(lines[i])
           && !(lines[i].trim().startsWith("|") && i + 1 < lines.length && isTableDivider(lines[i + 1]))) {
      para.push(lines[i]);
      i++;
    }
    if (para.length) {
      out.push(`<p>${renderInline(para.join(" "))}</p>`);
    }
  }

  return out.join("\n");
}

// ----------------------------------------------------------------------
// API helpers
// ----------------------------------------------------------------------
async function api(path, opts = {}) {
  // Use a relative URL so the request goes to wherever the SPA is
  // mounted (e.g. /job-search-agent/api/sessions on a path-prefixed
  // deploy, or /api/sessions at the domain root).
  // Stripping the leading slash is what makes it relative — the
  // browser resolves it against the current page's path.
  const url = path.startsWith("/") ? path.slice(1) : path;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function loadSessions() {
  state.sessions = await api("/api/sessions");
  renderSessionList();
}

async function createSession() {
  const s = await api("/api/sessions", { method: "POST", body: JSON.stringify({}) });
  state.currentSessionId = s.id;
  await loadSessions();
  openSession(s.id);
}

async function loadSession(id) {
  return api(`/api/sessions/${id}`);
}

async function renameCurrentSession() {
  if (!state.currentSessionId) return;
  const s = await loadSession(state.currentSessionId);
  const next = prompt("Rename session", s.title || "");
  if (!next || next === s.title) return;
  await api(`/api/sessions/${state.currentSessionId}`, {
    method: "PATCH",
    body: JSON.stringify({ title: next }),
  });
  $("chat-title").textContent = next;
  await loadSessions();
}

async function deleteCurrentSession() {
  if (!state.currentSessionId) return;
  if (!confirm("Delete this session? This cannot be undone.")) return;
  await api(`/api/sessions/${state.currentSessionId}`, { method: "DELETE" });
  state.currentSessionId = null;
  $("chat-title").textContent = "New research";
  $("messages").innerHTML = "";
  showWelcome(true);
  await loadSessions();
}

// ----------------------------------------------------------------------
// Rendering
// ----------------------------------------------------------------------
function renderSessionList() {
  const list = $("session-list");
  list.innerHTML = "";
  if (!state.sessions.length) {
    list.innerHTML = '<div class="empty">No research yet.<br>Click "+ New research" to start.</div>';
    return;
  }
  for (const s of state.sessions) {
    const el = document.createElement("a");
    el.className = "item" + (s.id === state.currentSessionId ? " active" : "");
    el.innerHTML = `${escapeHtml(s.title)}<span class="meta">${s.message_count} message${s.message_count === 1 ? "" : "s"}</span>`;
    el.onclick = () => openSession(s.id);
    list.appendChild(el);
  }
}

function showWelcome(show) {
  let el = $("welcome");
  if (show && !el) {
    // The welcome div lives inside #messages, which openSession wipes
    // when opening an existing session. Re-create it on demand.
    el = document.createElement("div");
    el.id = "welcome";
    el.className = "welcome";
    el.innerHTML = `
      <h2>What role are you trying to fill?</h2>
      <p>Tell me about a role and a location, and I'll research the market for you.</p>
      <p class="hint">Example: <em>"I need to hire a senior backend engineer in Berlin, hybrid, Go and Kubernetes."</em></p>
    `;
    $("messages").appendChild(el);
  }
  if (el) el.style.display = show ? "block" : "none";
}

function appendMessage(role, contentHtml, opts = {}) {
  showWelcome(false);
  const div = document.createElement("div");
  div.className = `message ${role}` + (opts.streaming ? " streaming" : "");
  const roleLabel = role === "user" ? "You" : "JobScope";
  div.innerHTML = `<span class="role">${roleLabel}</span><div class="content">${contentHtml}</div>`;
  $("messages").appendChild(div);
  $("messages").scrollTop = $("messages").scrollHeight;
  return div;
}

function appendActivity() {
  showWelcome(false);
  const div = document.createElement("div");
  div.className = "activity";
  div.innerHTML = "Researching… <span class=\"chips\"></span>";
  $("messages").appendChild(div);
  state.currentActivity = div;
  $("messages").scrollTop = $("messages").scrollHeight;
  return div;
}

function addSearchChip(query, active) {
  if (!state.currentActivity) return;
  const chip = document.createElement("span");
  chip.className = "search" + (active ? " active" : "");
  chip.textContent = query;
  state.currentActivity.querySelector(".chips").appendChild(chip);
}

function setStatus(text, isError = false) {
  const el = $("status-line");
  el.textContent = text || "";
  el.className = isError ? "error" : "";
}

function setBusy(busy) {
  $("send-btn").disabled = busy;
  $("input").disabled = busy;
  $("new-session").disabled = busy;
  document.querySelectorAll("#session-list .item").forEach((el) => { el.style.pointerEvents = busy ? "none" : ""; });
}

async function openSession(id) {
  if (state.inFlight) {
    if (!confirm("A research turn is in progress. Discard it and switch?")) return;
    state.inFlight.abort();
    state.inFlight = null;
  }

  // On mobile, opening a session from the sidebar should also
  // dismiss the overlay so the user sees the chat.
  if ($("sidebar").classList.contains("open")) {
    closeSidebar();
  }
  state.currentSessionId = id;
  const s = await loadSession(id);
  $("chat-title").textContent = s.title;
  const msgs = $("messages");
  msgs.innerHTML = "";
  state.currentActivity = null;
  if (!s.messages.length) {
    showWelcome(true);
  } else {
    for (const m of s.messages) {
      const html = m.role === "assistant" ? renderMarkdown(m.content) : escapeHtml(m.content).replace(/\n/g, "<br>");
      appendMessage(m.role, html);
    }
  }
  renderSessionList();
}

// ----------------------------------------------------------------------
// Send a turn
// ----------------------------------------------------------------------
async function sendMessage(text) {
  if (!text.trim()) return;
  if (!state.currentSessionId) {
    await createSession();
  }

  // Optimistic user bubble.
  appendMessage("user", escapeHtml(text).replace(/\n/g, "<br>"));
  const assistant = appendMessage("assistant", "", { streaming: true });
  const contentEl = assistant.querySelector(".content");
  let accumulated = "";
  let messageDone = false;

  setBusy(true);
  setStatus("Researching…");
  appendActivity();

  const controller = new AbortController();
  state.inFlight = controller;

  try {
    const res = await fetch("api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.currentSessionId, message: text }),
      signal: controller.signal,
    });
    if (!res.ok || !res.body) {
      const detail = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(detail.detail || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // SSE messages are separated by blank lines.
      let sep;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.slice(0, sep).trim();
        buffer = buffer.slice(sep + 2);
        if (!raw.startsWith("data:")) continue;
        const payload = raw.replace(/^data:\s*/, "");
        if (!payload) continue;
        let step;
        try { step = JSON.parse(payload); } catch { continue; }
        if (step.type === "search_start") {
          addSearchChip(step.query, true);
        } else if (step.type === "search_done") {
          // Mark the matching chip as completed.
          if (state.currentActivity) {
            const chips = state.currentActivity.querySelectorAll(".search");
            for (const c of chips) {
              if (c.textContent === step.query) { c.classList.remove("active"); break; }
            }
          }
        } else if (step.type === "text_delta") {
          accumulated += step.text || "";
          contentEl.innerHTML = renderMarkdown(accumulated);
          $("messages").scrollTop = $("messages").scrollHeight;
        } else if (step.type === "message_done") {
          messageDone = true;
          accumulated = step.text || accumulated;
          contentEl.innerHTML = renderMarkdown(accumulated);
        } else if (step.type === "error") {
          setStatus(step.message || "Error", true);
        } else if (step.type === "end") {
          // terminal marker
        }
      }
    }

    if (!messageDone) {
      // Stream ended without message_done — render whatever we accumulated.
      contentEl.innerHTML = renderMarkdown(accumulated);
    }
    assistant.classList.remove("streaming");
    setStatus("");
    await loadSessions();   // refresh title in sidebar
  } catch (err) {
    if (err.name === "AbortError") {
      setStatus("Cancelled.");
    } else {
      setStatus(`Error: ${err.message}`, true);
      contentEl.innerHTML = `<p><em>Request failed: ${escapeHtml(err.message)}</em></p>`;
    }
  } finally {
    state.inFlight = null;
    state.currentActivity = null;
    setBusy(false);
    $("input").focus();
  }
}

// ----------------------------------------------------------------------
// Export
// ----------------------------------------------------------------------
function exportSession() {
  if (!state.currentSessionId) {
    alert("Open a session first.");
    return;
  }
  // Build a standalone HTML doc with the current messages.
  const msgs = [...document.querySelectorAll("#messages .message")].map((el) => {
    const role = el.classList.contains("user") ? "You" : "JobScope";
    return `<section class="msg ${el.classList.contains("user") ? "user" : "assistant"}">
      <h3>${role}</h3>
      <div>${el.querySelector(".content").innerHTML}</div>
    </section>`;
  }).join("\n");
  const html = `<!doctype html><html><head><meta charset="utf-8">
<title>JobScope Export — ${escapeHtml($("chat-title").textContent)}</title>
<style>
  body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:780px;margin:40px auto;padding:0 20px;color:#222;line-height:1.5}
  h1{font-size:22px;margin-bottom:24px}
  .msg{border:1px solid #ddd;border-radius:8px;padding:16px;margin-bottom:16px}
  .msg.user{background:#f3f6ff}
  .msg.assistant{background:#fafafa}
  .msg h3{margin:0 0 8px 0;font-size:13px;color:#666;text-transform:uppercase;letter-spacing:.05em}
  table{border-collapse:collapse;margin:8px 0}
  th,td{border:1px solid #ccc;padding:6px 10px;text-align:left}
  th{background:#eee}
  code{background:#f4f4f4;padding:2px 5px;border-radius:4px;font-size:12.5px}
  pre{background:#f4f4f4;padding:12px;border-radius:6px;overflow-x:auto}
  blockquote{border-left:3px solid #4f8cff;margin:0;padding:4px 12px;color:#555}
  a{color:#1f5dd8}
</style>
</head><body>
<h1>${escapeHtml($("chat-title").textContent)}</h1>
${msgs || "<p><em>No messages.</em></p>"}
</body></html>`;
  const blob = new Blob([html], { type: "text/html" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `jobscope-${(state.currentSessionId || "export").slice(0, 8)}.html`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// ----------------------------------------------------------------------
// Health
// ----------------------------------------------------------------------
async function checkHealth() {
  try {
    const h = await api("/api/health");
    $("health").classList.add("ok");
    $("health").title = `Connected — model: ${h.model}`;
    $("model-name").textContent = h.model;
    if (!h.has_api_key) {
      $("health").classList.remove("ok");
      $("health").classList.add("error");
      $("health").title = "Server has no ANTHROPIC_API_KEY set";
    }
  } catch {
    $("health").classList.add("error");
    $("model-name").textContent = "Server unreachable";
  }
}

// ----------------------------------------------------------------------
// Mobile sidebar (hamburger menu)
// ----------------------------------------------------------------------
let _backdropEl = null;

function openSidebar() {
  $("sidebar").classList.add("open");
  $("menu-toggle").classList.add("open");
  $("menu-toggle").setAttribute("aria-label", "Close past research");
  if (!_backdropEl) {
    _backdropEl = document.createElement("div");
    _backdropEl.id = "backdrop";
    document.body.appendChild(_backdropEl);
    _backdropEl.addEventListener("click", closeSidebar);
  }
  // Force a reflow before adding .open so the transition fires.
  void _backdropEl.offsetWidth;
  _backdropEl.classList.add("open");
}

function closeSidebar() {
  $("sidebar").classList.remove("open");
  $("menu-toggle").classList.remove("open");
  $("menu-toggle").setAttribute("aria-label", "Show past research");
  if (_backdropEl) _backdropEl.classList.remove("open");
}

function toggleSidebar() {
  if ($("sidebar").classList.contains("open")) closeSidebar();
  else openSidebar();
}

// ----------------------------------------------------------------------
// Wire up
// ----------------------------------------------------------------------
window.addEventListener("DOMContentLoaded", () => {
  $("new-session").onclick = createSession;
  $("rename").onclick = renameCurrentSession;
  $("menu-toggle").onclick = toggleSidebar;

  // Add a delete button next to rename.
  const delBtn = document.createElement("button");
  delBtn.id = "delete";
  delBtn.title = "Delete session";
  delBtn.textContent = "🗑";
  delBtn.onclick = deleteCurrentSession;
  $("rename").insertAdjacentElement("afterend", delBtn);

  $("export").onclick = exportSession;

  const form = $("send-form");
  const input = $("input");

  form.onsubmit = (e) => {
    e.preventDefault();
    const text = input.value;
    input.value = "";
    autoSizeInput();
    sendMessage(text);
  };

  // Enter sends, Shift+Enter inserts newline.
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
  input.addEventListener("input", autoSizeInput);

  // Esc closes the mobile sidebar.
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && $("sidebar").classList.contains("open")) {
      closeSidebar();
    }
  });

  function autoSizeInput() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
  }

  checkHealth();
  loadSessions();
});
