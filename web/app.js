// RAG Chatbot web UI.
// Streams each answer over SSE, draws a per-question ribbon of the five
// pipeline stages sized by time spent, and manages documents with
// background indexing jobs.

const STAGES = [
  { id: "query_translation", label: "Translate", color: "var(--translation)" },
  { id: "routing", label: "Route", color: "var(--routing)" },
  { id: "query_construction", label: "Construct", color: "var(--construction)" },
  { id: "retrieval", label: "Retrieve", color: "var(--retrieval)" },
  { id: "generation", label: "Answer", color: "var(--generation)" },
];
const STAGE = Object.fromEntries(STAGES.map((s) => [s.id, s]));

const $ = (id) => document.getElementById(id);
const thread = $("thread");
const input = $("input");
const sendBtn = $("send");

let history = [];          // [{role, content}] for follow-up questions
let controller = null;     // AbortController of the answer being streamed
let documentsCount = 0;
const pollingJobs = new Set();

// ---------- small helpers ----------
const escapeHtml = (text) =>
  String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function seconds(ms) {
  if (ms == null) return "";
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(ms < 10000 ? 1 : 0)} s`;
}

function renderMarkdown(text) {
  const html = window.marked && window.DOMPurify
    ? DOMPurify.sanitize(marked.parse(text))
    : escapeHtml(text).replace(/\n/g, "<br>");
  return html.replace(/\[(\d{1,2})\](?![(:])/g, '<button type="button" class="cite" data-n="$1">$1</button>');
}

function codeify(message) {
  return escapeHtml(message).replace(/^(Text-to-SQL|Text-to-Cypher): ([\s\S]+)$/, "$1: <code>$2</code>");
}

function nearBottom() {
  return thread.scrollHeight - thread.scrollTop - thread.clientHeight < 160;
}

function scrollDown(force = false) {
  if (force || nearBottom()) thread.scrollTop = thread.scrollHeight;
}

// ---------- status and library ----------
function setState(kind, text) {
  const el = $("state");
  el.className = `state ${kind}`;
  el.querySelector("span").textContent = text;
}

async function refreshStatus() {
  try {
    const s = await (await fetch("/api/status")).json();
    document.title = s.title;
    $("title").textContent = s.title;
    $("f-strategy").textContent = `with ${s.strategy}`;
    $("f-routing").textContent = s.features.routing === "off" ? "off" : `${s.features.routing} routing`;
    $("f-retrieval").textContent = `${s.features.ranking} re-ranking${s.features.crag ? ", CRAG relevance check" : ""}`;
    $("f-generation").textContent = s.features.self_rag ? "with citations, checked by Self-RAG" : "with citations";
    $("f-stores").textContent =
      `${s.stores.vectorstore.toLocaleString()} passages, ${s.stores.relational_db} tables, ${s.stores.graph_db.toLocaleString()} graph facts`;
    const paused = (s.models || []).filter((m) => m.paused_for > 0);
    const pausedText = paused.map((m) => {
      const t = m.paused_for >= 3600 ? `${Math.round(m.paused_for / 3600)} h` : m.paused_for >= 60
        ? `${Math.round(m.paused_for / 60)} min` : `${m.paused_for} s`;
      return ` ${m.name.split(":").pop()} is paused for ${t} (${m.reason}).`;
    }).join("");
    $("f-llm").textContent = `Model: ${s.llm}${s.fallback ? `, backup ${s.fallback}` : ""}.${pausedText}`;
    $("files").accept = s.extensions.join(",");
    if (!s.ready) setState("bad", "Add a document to start");
    else if (s.jobs.length) setState("busy", "Indexing in the background");
    else setState("ok", "Ready");
    for (const job of s.jobs) if (!pollingJobs.has(job.id)) pollJob(job.id);
    return s;
  } catch {
    setState("bad", "Can't reach the server. Is app.py running?");
  }
}

async function refreshDocuments(jobsBySource = {}) {
  let docs = [];
  try {
    docs = await (await fetch("/api/documents")).json();
  } catch {
    return;
  }
  documentsCount = docs.length;
  const list = $("doc-list");
  list.innerHTML = "";
  $("doc-empty").hidden = docs.length > 0;
  for (const d of docs) list.appendChild(documentRow(d, jobsBySource[d.source]));
  for (const job of Object.values(pendingJobs)) {
    if (!docs.some((d) => job.sources?.includes(d.source))) list.prepend(jobRow(job));
  }
  updateWelcome();
}

function documentRow(doc, job) {
  const li = document.createElement("li");
  li.className = "doc";
  const counts = [
    doc.pages ? `${doc.pages} pages` : null,
    `${doc.chunks} passages`,
    doc.summaries ? `${doc.summaries} summaries` : null,
  ].filter(Boolean).join(", ");
  li.innerHTML = `
    <div class="doc-row"><span class="doc-name"></span>
      <button class="doc-remove" type="button" aria-label="Remove ${escapeHtml(doc.name)}">Remove</button></div>
    <div class="doc-meta">${escapeHtml(counts)}</div>`;
  li.querySelector(".doc-name").textContent = doc.name;
  li.querySelector(".doc-name").title = doc.source;
  if (job && job.state === "enriching") li.appendChild(progressEl(job, true));
  li.querySelector(".doc-remove").addEventListener("click", () => confirmRemove(li, doc));
  return li;
}

function jobRow(job) {
  const li = document.createElement("li");
  li.className = `doc ${job.state === "error" ? "error" : ""}`;
  li.innerHTML = `<div class="doc-row"><span class="doc-name"></span></div>`;
  li.querySelector(".doc-name").textContent = job.files.join(", ");
  if (job.state === "error") {
    li.insertAdjacentHTML("beforeend", `<div class="doc-meta">${escapeHtml(job.message)}</div>`);
  } else {
    li.appendChild(progressEl(job, false));
  }
  return li;
}

function progressEl(job, searchable) {
  const wrap = document.createElement("div");
  wrap.className = "progress";
  const pct = job.total ? Math.round((job.done / job.total) * 100) : 0;
  const count = job.total > 1 ? ` ${job.done} of ${job.total}` : "";
  wrap.innerHTML = `
    <div class="progress-bar ${job.total > 1 ? "" : "indeterminate"}"><i style="width:${pct}%"></i></div>
    <div class="progress-label ${searchable ? "searchable" : ""}">${escapeHtml(job.label)}${count}</div>`;
  return wrap;
}

function confirmRemove(li, doc) {
  if (li.querySelector(".doc-confirm")) return;
  const box = document.createElement("div");
  box.className = "doc-confirm";
  box.innerHTML = `<span>Remove from every store?</span>
    <button type="button" class="danger">Remove</button><button type="button">Keep</button>`;
  const [removeBtn, keepBtn] = box.querySelectorAll("button");
  keepBtn.addEventListener("click", () => box.remove());
  removeBtn.addEventListener("click", async () => {
    removeBtn.disabled = true;
    const res = await fetch(`/api/documents?source=${encodeURIComponent(doc.source)}`, { method: "DELETE" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      box.innerHTML = `<span>${escapeHtml(err.detail || "Couldn't remove it.")}</span>`;
      return;
    }
    $("upload-note").textContent = `Removed ${doc.name}.`;
    refreshDocuments();
    refreshStatus();
  });
  li.appendChild(box);
  keepBtn.focus();
}

// ---------- uploads and job polling ----------
const pendingJobs = {};   // job id -> job, while not yet listed as a document

async function upload(fileList) {
  const files = [...fileList];
  if (!files.length) return;
  const form = new FormData();
  for (const f of files) form.append("files", f);
  $("upload-note").textContent = "";
  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed.");
    if (data.skipped?.length) $("upload-note").textContent = `Skipped unsupported files: ${data.skipped.join(", ")}`;
    pendingJobs[data.job.id] = data.job;
    refreshDocuments();
    pollJob(data.job.id);
  } catch (e) {
    $("upload-note").textContent = e.message;
  }
  $("files").value = "";
}

async function pollJob(id) {
  pollingJobs.add(id);
  let lastState = "";
  for (;;) {
    let job;
    try {
      job = await (await fetch(`/api/jobs/${id}`)).json();
    } catch {
      await new Promise((r) => setTimeout(r, 2000));
      continue;
    }
    if (job.state === "indexing" || job.state === "queued") {
      pendingJobs[id] = job;
      refreshList(job);
    } else {
      delete pendingJobs[id];
      if (job.state !== lastState) {
        await refreshDocuments({ ...Object.fromEntries((job.sources || []).map((s) => [s, job])) });
        refreshStatus();
      } else {
        refreshList(job);
      }
    }
    lastState = job.state;
    if (job.state === "done" || job.state === "error") {
      pollingJobs.delete(id);
      if (job.state === "error") {
        pendingJobs[id] = job;
        refreshDocuments();
      } else {
        $("upload-note").textContent = `${job.files.join(", ")} is fully indexed.`;
      }
      return;
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
}

let listTimer = null;
function refreshList(job) {
  // Update just the progress text/bar without refetching every second.
  const rows = [...document.querySelectorAll("#doc-list .doc")];
  const row = rows.find((r) => {
    const name = r.querySelector(".doc-name")?.textContent || "";
    return job.files.some((f) => name.includes(f));
  });
  if (!row) {
    clearTimeout(listTimer);
    listTimer = setTimeout(() => refreshDocuments(), 300);
    return;
  }
  row.querySelector(".progress")?.replaceWith(progressEl(job, job.state === "enriching"));
  if (!row.querySelector(".progress")) row.appendChild(progressEl(job, job.state === "enriching"));
}

// ---------- the conversation ----------
function updateWelcome() {
  const welcome = $("welcome");
  if (!welcome) return;
  const empty = documentsCount === 0 && !Object.keys(pendingJobs).length;
  $("dropzone").hidden = !empty;
  $("suggestions").hidden = empty;
  $("welcome-title").textContent = empty ? "Start by adding a document" : "Ask your documents anything";
}

function addTurn(question) {
  $("welcome")?.remove();
  const turn = document.createElement("article");
  turn.className = "turn";
  turn.innerHTML = `
    <h3 class="question"></h3>
    <div class="ribbon">
      <button type="button" class="ribbon-bar" aria-label="Show how this answer was found">
        ${STAGES.map((s) => `<span class="pending" data-stage="${s.id}" style="--c:${s.color}; flex-grow:1"></span>`).join("")}
      </button>
      <button type="button" class="ribbon-caption" aria-expanded="false">
        <span class="now">Planning the search…</span><span class="time"></span>
      </button>
      <ol class="steps" hidden></ol>
    </div>
    <p class="notice" hidden></p>
    <div class="answer typing"></div>
    <div class="error-slot"></div>
    <div class="answer-actions" hidden>
      <button type="button" class="copy">Copy answer</button>
      <span class="model"></span>
    </div>
    <div class="sources" hidden></div>`;
  turn.querySelector(".question").textContent = question;
  const toggle = () => {
    const steps = turn.querySelector(".steps");
    steps.hidden = !steps.hidden;
    turn.querySelector(".ribbon-caption").setAttribute("aria-expanded", String(!steps.hidden));
  };
  turn.querySelector(".ribbon-bar").addEventListener("click", toggle);
  turn.querySelector(".ribbon-caption").addEventListener("click", toggle);
  thread.appendChild(turn);
  scrollDown(true);
  return turn;
}

function ribbonState(turn, timings, activeStage) {
  const total = Object.values(timings).reduce((a, b) => a + b, 0) || 1;
  for (const s of STAGES) {
    const seg = turn.querySelector(`.ribbon-bar [data-stage="${s.id}"]`);
    const ms = timings[s.id];
    seg.classList.toggle("pending", ms == null && s.id !== activeStage);
    seg.classList.toggle("active", s.id === activeStage);
    // Width shows where the time went; a sliver keeps quick stages visible.
    seg.style.flexGrow = ms == null ? (s.id === activeStage ? 0.6 : 0.4) : Math.max(ms / total, 0.03) * 10;
  }
}

function addStep(turn, ev, timings) {
  const stage = STAGE[ev.stage] || { label: ev.stage, color: "var(--muted)" };
  timings[ev.stage] = (timings[ev.stage] || 0) + (ev.ms || 0);
  const li = document.createElement("li");
  li.innerHTML = `<span class="stage" style="--c:${stage.color}">${stage.label}</span>
    <span class="text">${codeify(ev.message)}</span><span class="ms">${ev.ms ? seconds(ev.ms) : ""}</span>`;
  turn.querySelector(".steps").appendChild(li);
  turn.querySelector(".now").textContent = ev.message;
  const next = STAGES[STAGES.findIndex((s) => s.id === ev.stage) + 1];
  ribbonState(turn, timings, ev.stage === "retrieval" ? "generation" : next?.id);
}

function renderSources(turn, sources) {
  const box = turn.querySelector(".sources");
  box.innerHTML = "";
  box.hidden = !sources.length;
  for (const s of sources) {
    const row = document.createElement("div");
    row.className = "source";
    row.dataset.n = s.n;
    const label = s.url
      ? `<a href="${escapeHtml(s.url)}" target="_blank" rel="noopener">${escapeHtml(s.label)}</a>`
      : escapeHtml(s.label);
    row.innerHTML = `<span class="n">${s.n}</span>
      <div><div class="label">${label}</div><div class="snippet">${escapeHtml(s.snippet)}</div></div>
      <button type="button" class="more">Show passage</button>`;
    const more = row.querySelector(".more");
    more.addEventListener("click", () => {
      row.classList.toggle("open");
      more.textContent = row.classList.contains("open") ? "Show less" : "Show passage";
    });
    box.appendChild(row);
  }
  turn.querySelectorAll(".cite").forEach((c) =>
    c.addEventListener("click", () => {
      const row = box.querySelector(`[data-n="${c.dataset.n}"]`);
      if (!row) return;
      row.scrollIntoView({ behavior: "smooth", block: "nearest" });
      row.classList.add("flash");
      setTimeout(() => row.classList.remove("flash"), 1400);
    })
  );
}

function setBusy(busy) {
  sendBtn.textContent = busy ? "Stop" : "Ask";
  sendBtn.classList.toggle("stop", busy);
  sendBtn.type = busy ? "button" : "submit";
}

async function ask(question) {
  if (controller || !question.trim()) return;
  const turn = addTurn(question);
  const answerEl = turn.querySelector(".answer");
  const timings = {};
  let answer = "";
  let finished = false;
  controller = new AbortController();
  setBusy(true);
  ribbonState(turn, timings, "query_translation");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question, history }),
      signal: controller.signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "The server couldn't answer. Please try again.");
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        if (!chunk.startsWith("data: ")) continue;
        const ev = JSON.parse(chunk.slice(6));
        if (ev.kind === "step") {
          addStep(turn, ev, timings);
        } else if (ev.kind === "token") {
          if (!answer) turn.querySelector(".now").textContent = "Writing the answer…";
          answer += ev.text;
          answerEl.innerHTML = renderMarkdown(answer);
          scrollDown();
        } else if (ev.kind === "reset") {
          answer = "";
          answerEl.innerHTML = "";
          const notice = turn.querySelector(".notice");
          notice.hidden = false;
          notice.textContent = "The first draft wasn't fully supported by the documents, so it's being rewritten.";
        } else if (ev.kind === "done") {
          finished = true;
          answer = ev.answer;
          answerEl.innerHTML = renderMarkdown(answer || "No answer was produced.");
          const before = Object.values(timings).reduce((a, b) => a + b, 0);
          timings.generation = (timings.generation || 0) + Math.max(ev.ms - before, 0);
          ribbonState(turn, timings, null);
          turn.querySelector(".now").textContent = `Answered from ${ev.sources.length} source${ev.sources.length === 1 ? "" : "s"}`;
          turn.querySelector(".time").textContent =
            `${seconds(ev.ms)} total${ev.first_token_ms ? `, first words after ${seconds(ev.first_token_ms)}` : ""}`;
          turn.querySelector(".model").textContent = ev.model ? `Answered by ${ev.model.split(":").pop()}` : "";
          turn.querySelector(".answer-actions").hidden = false;
          renderSources(turn, ev.sources);
        } else if (ev.kind === "error") {
          throw new Error(ev.message);
        }
      }
    }
    if (finished) history.push({ role: "user", content: question }, { role: "assistant", content: answer });
  } catch (e) {
    const stopped = e.name === "AbortError";
    ribbonState(turn, timings, null);
    turn.querySelector(".now").textContent = stopped ? "Stopped" : "Didn't finish";
    const slot = turn.querySelector(".error-slot");
    slot.innerHTML = `<p class="error-note"><span></span><button type="button">Ask again</button></p>`;
    slot.querySelector("span").textContent = stopped ? "You stopped this answer." : e.message;
    slot.querySelector("button").addEventListener("click", () => {
      turn.remove();
      ask(question);
    });
    if (!answer) answerEl.remove();
  } finally {
    answerEl.classList.remove("typing");
    controller = null;
    setBusy(false);
    input.focus();
    refreshStatus();
  }
}

// ---------- wiring ----------
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value;
  if (!text.trim()) return;
  input.value = "";
  input.style.height = "auto";
  ask(text);
});
sendBtn.addEventListener("click", () => {
  if (controller) controller.abort();
});
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!controller) $("composer").requestSubmit();
  }
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 200)}px`;
});
document.querySelectorAll(".suggestions button").forEach((b) => b.addEventListener("click", () => ask(b.textContent)));

$("new-chat").addEventListener("click", () => {
  if (controller) controller.abort();
  history = [];
  thread.innerHTML = `<section class="welcome" id="welcome">
      <h2 id="welcome-title">Ask your documents anything</h2>
      <p>Answers quote the passages they come from, and the coloured bar under each question shows how the answer was found.</p>
      <div class="suggestions" id="suggestions"></div>
      <label class="dropzone" id="dropzone" for="files" hidden><b>Add a document to start</b>
        <span>Drop files here or choose them. PDF, DOCX, CSV, Markdown, TXT and HTML work.</span></label>
    </section>`;
  for (const q of ["What is this document about?", "Summarize the key points.", "How are the main ideas related?"]) {
    const b = document.createElement("button");
    b.textContent = q;
    b.addEventListener("click", () => ask(q));
    $("suggestions").appendChild(b);
  }
  updateWelcome();
  input.focus();
});

$("files").addEventListener("change", (e) => upload(e.target.files));
$("open-library").addEventListener("click", () => $("library").classList.add("open"));
$("close-library").addEventListener("click", () => $("library").classList.remove("open"));

// Drop files anywhere on the page.
let dragDepth = 0;
window.addEventListener("dragenter", (e) => {
  if (![...e.dataTransfer.types].includes("Files")) return;
  dragDepth++;
  $("drag-veil").hidden = false;
});
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) $("drag-veil").hidden = true;
});
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  $("drag-veil").hidden = true;
  if (e.dataTransfer.files.length) upload(e.dataTransfer.files);
});

refreshStatus();
refreshDocuments();
input.focus();

// ?q=... asks that question on load, so a question can be shared as a link.
const initial = new URLSearchParams(location.search).get("q");
if (initial) ask(initial);
