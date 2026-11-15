// RAG Chatbot web UI: streams pipeline steps and the answer over SSE.

const STAGES = {
  query_translation: { label: "Translation", color: "var(--translation)" },
  routing: { label: "Routing", color: "var(--routing)" },
  query_construction: { label: "Construction", color: "var(--construction)" },
  retrieval: { label: "Retrieval", color: "var(--retrieval)" },
  generation: { label: "Generation", color: "var(--generation)" },
};
const ORDER = Object.keys(STAGES);

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const input = $("input");
const sendBtn = $("send");
let history = [];   // [{role, content}] sent back to the server for follow-ups
let busy = false;

// ---------- helpers ----------
function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderMarkdown(text) {
  let html;
  if (window.marked && window.DOMPurify) {
    html = DOMPurify.sanitize(marked.parse(text));
  } else {
    html = escapeHtml(text).replace(/\n/g, "<br>");
  }
  // [1] / [2][3] citations → clickable chips (outside code blocks is good enough here)
  return html.replace(/\[(\d{1,2})\](?!\()/g, '<span class="cite" data-n="$1">$1</span>');
}

function withCode(text) {
  // Show SQL / Cypher / queries in monospace inside trace lines.
  return escapeHtml(text).replace(/^(Text-to-SQL|Text-to-Cypher): (.+)$/, "$1: <code>$2</code>");
}

function scrollDown() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ---------- status ----------
async function refreshStatus(statusData) {
  try {
    const s = statusData || (await (await fetch("/api/status")).json());
    document.title = s.title;
    $("title").textContent = s.title;
    $("description").textContent = s.description;
    for (const [k, v] of Object.entries(s.stores)) $(`s-${k}`).textContent = v.toLocaleString();
    $("f-strategy").textContent = s.strategy;
    $("f-routing").textContent = s.features.routing;
    $("f-indexing").textContent = s.features.indexing;
    $("f-retrieval").textContent = `rank: ${s.features.ranking}` + (s.features.crag ? " · CRAG" : "");
    $("f-generation").textContent = s.features.self_rag ? "grounded answers · Self-RAG" : "grounded answers";
    $("f-llm").textContent = `LLM: ${s.llm}` + (s.fallback ? ` → ${s.fallback} fallback` : "");
    $("exts").textContent = s.extensions.join(" ");
    $("ready").textContent = s.ready ? "Ready" : "No documents yet: upload some to start";
  } catch {
    $("ready").textContent = "Server unreachable";
  }
}

// ---------- messages ----------
function addUserMessage(text) {
  $("empty")?.remove();
  const el = document.createElement("div");
  el.className = "msg user";
  el.innerHTML = `<div class="bubble"></div>`;
  el.querySelector(".bubble").textContent = text;
  messagesEl.appendChild(el);
}

function addAssistantMessage() {
  const el = document.createElement("div");
  el.className = "msg assistant";
  el.innerHTML = `
    <details class="trace" open>
      <summary><span class="bar">${ORDER.map((s) => `<span data-stage="${s}"></span>`).join("")}</span>
        <span class="label">Running the pipeline…</span></summary>
      <ol></ol>
    </details>
    <div class="notice" hidden></div>
    <div class="answer cursor"></div>
    <div class="sources"></div>`;
  messagesEl.appendChild(el);
  scrollDown();
  return el;
}

function addStep(el, ev) {
  const stage = STAGES[ev.stage] || { label: ev.stage, color: "var(--muted)" };
  const li = document.createElement("li");
  li.innerHTML = `<span class="stage" style="--c:${stage.color}">${stage.label}</span><span class="text">${withCode(ev.message)}</span>`;
  el.querySelector(".trace ol").appendChild(li);
  const seg = el.querySelector(`.bar [data-stage="${ev.stage}"]`);
  if (seg) seg.style.background = stage.color;
  el.querySelector(".trace .label").textContent = ev.message;
  scrollDown();
}

function renderSources(el, sources) {
  const box = el.querySelector(".sources");
  box.innerHTML = "";
  for (const s of sources) {
    const card = document.createElement("div");
    card.className = "source";
    card.dataset.n = s.n;
    const label = s.url ? `<a href="${escapeHtml(s.url)}" target="_blank" rel="noopener">${escapeHtml(s.label)}</a>` : escapeHtml(s.label);
    card.innerHTML = `<header><b>[${s.n}]</b><span>${label}</span></header><p>${escapeHtml(s.snippet)}</p>`;
    card.addEventListener("click", () => card.classList.toggle("open"));
    box.appendChild(card);
  }
  el.querySelectorAll(".cite").forEach((c) =>
    c.addEventListener("click", () => {
      const card = box.querySelector(`[data-n="${c.dataset.n}"]`);
      if (!card) return;
      card.scrollIntoView({ behavior: "smooth", block: "nearest" });
      card.classList.add("flash", "open");
      setTimeout(() => card.classList.remove("flash"), 1200);
    })
  );
}

// ---------- chat ----------
async function send(text) {
  if (busy || !text.trim()) return;
  busy = true;
  sendBtn.disabled = true;
  addUserMessage(text);
  const el = addAssistantMessage();
  const answerEl = el.querySelector(".answer");
  let answer = "";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, history }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        if (!chunk.startsWith("data: ")) continue;
        const ev = JSON.parse(chunk.slice(6));

        if (ev.kind === "step") addStep(el, ev);
        else if (ev.kind === "token") {
          answer += ev.text;
          answerEl.innerHTML = renderMarkdown(answer);
          scrollDown();
        } else if (ev.kind === "reset") {
          answer = "";
          answerEl.innerHTML = "";
          const notice = el.querySelector(".notice");
          notice.hidden = false;
          notice.textContent = "Self-RAG rejected the first draft, so it's trying again…";
        } else if (ev.kind === "done") {
          answer = ev.answer;
          answerEl.innerHTML = renderMarkdown(answer);
          renderSources(el, ev.sources);
          const steps = el.querySelectorAll(".trace li").length;
          el.querySelector(".trace .label").textContent = `Pipeline trace · ${steps} steps`;
          el.querySelector(".trace").open = false;
        } else if (ev.kind === "error") {
          throw new Error(ev.message);
        }
      }
    }
    history.push({ role: "user", content: text }, { role: "assistant", content: answer });
  } catch (e) {
    answerEl.innerHTML += `<p class="error">Error: ${escapeHtml(e.message)}</p>`;
  } finally {
    answerEl.classList.remove("cursor");
    busy = false;
    sendBtn.disabled = false;
    input.focus();
    refreshStatus();
  }
}

// ---------- upload ----------
async function upload(fileList) {
  if (!fileList.length) return;
  const status = $("upload-status");
  const form = new FormData();
  for (const f of fileList) form.append("files", f);
  status.textContent = `Indexing ${fileList.length} file(s)… this can take a while (graph + RAPTOR).`;
  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");
    status.textContent = data.message;
    refreshStatus(data.status);
  } catch (e) {
    status.textContent = e.message;
  }
}

// ---------- wiring ----------
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value;
  input.value = "";
  input.style.height = "auto";
  send(text);
});
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("composer").requestSubmit();
  }
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
});
document.querySelectorAll(".suggestions button").forEach((b) => b.addEventListener("click", () => send(b.textContent)));
$("clear").addEventListener("click", () => {
  history = [];
  messagesEl.innerHTML = "";
  input.focus();
});
$("menu").addEventListener("click", () => $("sidebar").classList.toggle("open"));

const drop = $("drop");
$("files").addEventListener("change", (e) => upload(e.target.files));
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  upload(e.dataTransfer.files);
});

refreshStatus();
