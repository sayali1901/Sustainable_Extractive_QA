"""Local web UI for document questions."""

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .local_llm import LocalLLM
from .rag_pipeline import LocalRAGPipeline


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SQUAD_DEV_PATH = os.path.join(PROJECT_ROOT, "data", "dev-v2.0.json")

# Fallback when a request does not select a prompt.
DEFAULT_PROMPT_ID = "rag_cited_answer"
PROMPT_IDS = {"rag_cited_answer", "rag_strict", "graph_answer"}
TOP_K_VALUES = {1, 3, 5}
WINDOW_VALUES = {100, 400}
OVERLAP_VALUES = {20, 80}
LATENCY_VALUES = {1500, 2500, 5000, 10000, 60000}
CONTEXT_VALUES = {900, 4000}


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sustainable Extractive QA</title>
  <style>
    :root {
      --bg: #f5f6f4;
      --surface: #ffffff;
      --ink: #1a1d19;
      --muted: #6b7168;
      --subtle: #9aa097;
      --line: #e2e4de;
      --line-soft: #eef0ea;
      --accent: #1f7a5a;
      --accent-dark: #155f46;
      --accent-soft: #e6f2ec;
      --warn: #a15c00;
      --warn-soft: #fbf1e3;
      --info: #2a5db0;
      --info-soft: #e6ecf7;
      --mono: ui-monospace, "SF Mono", "JetBrains Mono", Consolas, monospace;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
      font-size: 15px;
      line-height: 1.5;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 28px;
      border-bottom: 1px solid var(--line);
      background: var(--surface);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    header h1 {
      font-size: 17px;
      margin: 0;
      font-weight: 700;
      letter-spacing: -0.01em;
    }
    header .subtitle {
      font-size: 13px;
      color: var(--muted);
      margin-top: 2px;
    }
    .status {
      display: flex;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
    }
    .dot {
      width: 9px; height: 9px;
      border-radius: 50%;
      background: var(--warn);
    }
    .dot.ok { background: var(--accent); }

    .intro {
      display: grid;
      grid-template-columns: minmax(280px, 1fr) minmax(520px, 1.5fr);
      gap: 24px;
      align-items: center;
      margin-bottom: 18px;
    }
    .intro h2 {
      margin: 0 0 8px;
      font-size: 24px;
      line-height: 1.25;
      letter-spacing: -0.02em;
    }
    .intro p {
      margin: 0;
      color: var(--muted);
      max-width: 680px;
    }
    .method {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-top: 0;
    }
    .method-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.65);
      padding: 10px 12px;
    }
    .method-item strong {
      display: block;
      color: var(--ink);
      font-size: 13px;
    }
    .method-item span {
      color: var(--muted);
      font-size: 12px;
    }

    main {
      max-width: 1440px;
      margin: 0 auto;
      padding: 22px 24px 36px;
    }
    .workspace {
      display: grid;
      grid-template-columns: minmax(480px, 0.9fr) minmax(520px, 1.25fr);
      gap: 20px;
      align-items: start;
    }
    .control-panel .card:last-child {
      margin-bottom: 0;
    }
    #answerCard {
      position: sticky;
      top: 84px;
      min-height: 570px;
      max-height: calc(100vh - 106px);
      overflow-y: auto;
      margin-bottom: 0;
    }
    .card {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 20px;
    }
    label {
      display: block;
      font-size: 13px;
      font-weight: 600;
      color: var(--muted);
      margin-bottom: 8px;
    }
    textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-size: 15px;
      padding: 12px 14px;
      min-height: 90px;
      line-height: 1.5;
      resize: vertical;
      outline: none;
      transition: border-color 0.12s, box-shadow 0.12s;
    }
    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(31, 122, 90, 0.12);
    }
    .examples {
      margin-top: 12px;
    }
    .examples-label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 7px;
    }
    .example-list {
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
    }
    button.example-question {
      min-height: 0;
      border-color: var(--line);
      background: var(--accent-soft);
      color: var(--accent-dark);
      padding: 6px 9px;
      font-size: 12px;
      font-weight: 500;
      text-align: left;
    }
    button.example-question:hover:not(:disabled) { background: #dceee7; }
    .actions {
      display: flex;
      gap: 10px;
      margin-top: 14px;
    }
    button {
      min-height: 44px;
      border: 1px solid var(--accent);
      background: var(--accent);
      color: white;
      border-radius: 8px;
      padding: 0 22px;
      font: inherit;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.12s;
    }
    button:hover:not(:disabled) { background: var(--accent-dark); }
    button.secondary {
      background: #fff;
      color: var(--accent-dark);
    }
    button.secondary:hover:not(:disabled) { background: var(--accent-soft); }
    button:disabled { cursor: wait; opacity: 0.6; }
    .hint {
      font-size: 12px;
      color: var(--subtle);
      margin-top: 10px;
    }
    .privacy-note {
      color: var(--accent-dark);
      font-size: 12px;
      margin-top: 10px;
    }
    .settings-title {
      margin: 0 0 4px;
      font-size: 16px;
    }
    .settings-note {
      margin: 0 0 18px;
      color: var(--muted);
      font-size: 13px;
    }
    .settings-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }
    .field label {
      margin-bottom: 6px;
    }
    select, input[type="number"] {
      width: 100%;
      height: 42px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      padding: 0 10px;
    }
    select:focus, input[type="number"]:focus {
      border-color: var(--accent);
      outline: none;
      box-shadow: 0 0 0 3px rgba(31, 122, 90, 0.12);
    }
    .toggles {
      display: flex;
      flex-wrap: wrap;
      gap: 12px 24px;
      margin-top: 18px;
      padding-top: 16px;
      border-top: 1px solid var(--line-soft);
    }
    .toggle {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--ink);
      font-size: 13px;
      font-weight: 600;
      margin: 0;
    }
    .toggle input { width: 17px; height: 17px; accent-color: var(--accent); }

    .answer-empty {
      color: var(--subtle);
      font-size: 15px;
      padding: 24px 0;
      text-align: center;
    }
    .answer-text {
      font-size: 19px;
      line-height: 1.55;
      color: var(--ink);
      white-space: pre-wrap;
      margin: 0 0 16px 0;
    }
    .status-line {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 7px;
      font-size: 13px;
      font-weight: 500;
      margin-top: 14px;
    }
    .status-line.generated {
      background: var(--accent-soft);
      color: var(--accent-dark);
    }
    .status-line.retrieval {
      background: var(--info-soft);
      color: var(--info);
    }
    .status-line.error {
      background: var(--warn-soft);
      color: var(--warn);
    }
    .status-icon { font-size: 12px; }

    .citations-title {
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 12px;
    }
    .citation {
      border: 1px solid var(--line-soft);
      border-radius: 8px;
      padding: 12px 14px;
      margin-bottom: 10px;
      background: #fafbf8;
      font-size: 13.5px;
      line-height: 1.5;
    }
    .citation.selected {
      border-color: var(--accent);
      background: var(--accent-soft);
    }
    .citation-head {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 6px;
      font-size: 12px;
    }
    .citation-label {
      font-weight: 700;
      color: var(--accent-dark);
      font-family: var(--mono);
    }
    .citation-source {
      color: var(--muted);
    }
    .citation-badge {
      margin-left: auto;
      font-size: 10.5px;
      font-weight: 700;
      color: var(--accent-dark);
      background: #fff;
      border: 1px solid var(--accent);
      padding: 2px 7px;
      border-radius: 999px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .citation-text {
      color: var(--ink);
      max-height: 80px;
      overflow: hidden;
    }
    .citation-text.expanded { max-height: none; }
    .citation-toggle {
      background: none;
      border: none;
      color: var(--accent-dark);
      font-size: 12px;
      font-weight: 600;
      padding: 6px 0 0 0;
      cursor: pointer;
      min-height: 0;
    }
    .citation-toggle:hover { background: none; text-decoration: underline; }

    @media (max-width: 1050px) {
      .intro { grid-template-columns: 1fr; }
      .method { margin-top: 4px; }
      .workspace { grid-template-columns: 1fr; }
      #answerCard {
        position: static;
        min-height: 360px;
        max-height: none;
      }
    }
    @media (max-width: 700px) {
      main { padding: 16px; }
      header { padding: 12px 16px; }
      .card { padding: 18px; }
      .actions { flex-direction: column; }
      .method { grid-template-columns: 1fr; }
      .settings-grid { grid-template-columns: 1fr; }
      header { align-items: flex-start; gap: 12px; }
      .status { text-align: right; }
      button { width: 100%; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Sustainable Extractive QA</h1>
      <div class="subtitle">Local question answering with cited evidence</div>
    </div>
  </header>

  <main>
    <section class="intro">
      <h2>Ask the SQuAD research corpus</h2>
      <p>This interface tests a CPU-friendly retrieval pipeline with optional local generation. Answers stay grounded in the indexed source passages.</p>
      <div class="method">
        <div class="method-item">
          <strong>Local first</strong>
          <span>Documents remain on this machine</span>
        </div>
        <div class="method-item">
          <strong>Evidence based</strong>
          <span>Retrieved passages are shown below each answer</span>
        </div>
        <div class="method-item">
          <strong>Fallback ready</strong>
          <span>Extractive QA works without a local model</span>
        </div>
      </div>
    </section>

    <div class="workspace">
      <section class="control-panel">
        <div class="card">
          <label for="question">Question</label>
          <textarea id="question" placeholder="Ask a question about the indexed SQuAD passages…"></textarea>
          <div class="examples">
            <div class="examples-label">Try an example:</div>
            <div class="example-list" id="exampleQuestions"></div>
          </div>
          <div class="actions">
            <button id="askBtn">Ask</button>
            <button id="summaryBtn" class="secondary">Summarize evidence</button>
          </div>
          <div class="hint">Press Ctrl+Enter (or ⌘+Enter) to ask.</div>
          <div class="privacy-note">Processing is local. Ollama is contacted only through its local API.</div>
        </div>

        <div class="card">
          <h3 class="settings-title">Experiment settings</h3>
          <p class="settings-note">Change one setting at a time to compare retrieval and generation behaviour.</p>
          <div class="settings-grid">
            <div class="field">
              <label for="model">Local model</label>
              <select id="model"><option value="">Automatic</option></select>
            </div>
            <div class="field">
              <label for="promptId">Prompt</label>
              <select id="promptId">
                <option value="rag_cited_answer">Cited answer</option>
                <option value="rag_strict">Strict RAG</option>
              </select>
            </div>
            <div class="field">
              <label for="topK">Retrieved chunks (K)</label>
              <select id="topK">
                <option value="1">1</option>
                <option value="3" selected>3</option>
                <option value="5">5</option>
              </select>
            </div>
            <div class="field">
              <label for="latencyTarget">Latency budget</label>
              <select id="latencyTarget">
                <option value="1500">1.5 seconds</option>
                <option value="2500">2.5 seconds</option>
                <option value="5000" selected>5 seconds</option>
                <option value="10000">10 seconds</option>
                <option value="60000">No practical limit</option>
              </select>
            </div>
            <div class="field">
              <label for="windowSize">Chunk window</label>
              <select id="windowSize">
                <option value="100" selected>100 tokens</option>
                <option value="400">400 tokens</option>
              </select>
            </div>
            <div class="field">
              <label for="overlap">Chunk overlap</label>
              <select id="overlap">
                <option value="20" selected>20 tokens</option>
                <option value="80">80 tokens</option>
              </select>
            </div>
            <div class="field">
              <label for="contextTokens">Context budget</label>
              <select id="contextTokens">
                <option value="900">900 tokens</option>
                <option value="4000" selected>4,000 tokens</option>
              </select>
            </div>
          </div>
          <div class="toggles">
            <label class="toggle"><input id="useGenerator" type="checkbox" checked> Use local model</label>
            <label class="toggle"><input id="selfCorrect" type="checkbox"> Run self-correction</label>
          </div>
        </div>
      </section>

      <section class="card" id="answerCard">
        <div id="answerContainer">
          <div class="answer-empty">The answer and supporting passages will appear here.</div>
        </div>
      </section>
    </div>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const askBtn = $("askBtn");
    const summaryBtn = $("summaryBtn");
    const useGenerator = $("useGenerator");
    const sampleQuestions = [
      "What does a U.S. census report state that even after other factors there still exists this between earnings of men and women?",
      'Who is the author of the book, "Knowledge and Decisions"?',
      "What is Southern California often abbreviated as?",
      "What is the atomic number of oxygen?",
      "When did the oil crisis begin?"
    ];

    sampleQuestions.forEach((question) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "example-question";
      button.textContent = question;
      button.addEventListener("click", () => {
        $("question").value = question;
        $("question").focus();
      });
      $("exampleQuestions").appendChild(button);
    });

    async function refreshStatus() {
      // Check whether Ollama is ready.
      try {
        const res = await fetch("/api/status");
        const data = await res.json();
        // Populate the model selector with installed Ollama models.
        const modelSelect = $("model");
        const currentModel = modelSelect.value;
        data.local_model.installed_models.forEach((name) => {
          if (![...modelSelect.options].some((option) => option.value === name)) {
            modelSelect.add(new Option(name, name));
          }
        });
        modelSelect.value = currentModel;
      } catch {
        // The model list remains unchanged if status is unavailable.
      }
    }

    function setBusy(busy) {
      // Lock both actions during a request.
      askBtn.disabled = busy;
      summaryBtn.disabled = busy;
      if (busy) {
        $("answerContainer").innerHTML =
          '<div class="answer-empty">Working…</div>';
      }
    }

    function formatLatency(ms) {
      // Use units that are easy to read.
      if (ms == null) return "";
      if (ms < 1) return "<1 ms";
      if (ms < 1000) return `${Math.round(ms)} ms`;
      return `${(ms / 1000).toFixed(1)} s`;
    }
    function escapeHtml(s) {
      // Escape text before adding it to the page.
      return String(s ?? "").replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[c]));
    }

    function statusLine(data) {
      // Show which answer path was used.
      const total = formatLatency(data.latency_ms);
      if (data.generated) {
        const model = data.model || "local model";
        return `<div class="status-line generated">
          <span class="status-icon">●</span>
          Answered by ${escapeHtml(model)} in ${total}
        </div>`;
      }
      const why = data.fallback_reason === "no_local_model_available"
        // Convert internal reasons into readable labels.
        ? "no local model running"
        : data.fallback_reason === "generator_disabled"
          ? "generator disabled"
          : "retrieval path";
      return `<div class="status-line retrieval">
        <span class="status-icon">●</span>
        Answered from retrieved evidence in ${total} (${escapeHtml(why)})
      </div>`;
    }

    function settingsLine(data) {
      // Show the configuration used for this result.
      const path = data.generated ? "local generation" : "extractive fallback";
      return `<div class="hint">
        ${escapeHtml(path)} · ${escapeHtml(data.prompt_id)} · K=${data.top_k} ·
        chunks ${data.window_size}/${data.overlap} · context ${data.max_context_tokens}
      </div>`;
    }

    function citations(data) {
      // Mark citations used by the fallback answer.
      const list = data.citations || [];
      if (!list.length) return "";
      const selected = new Set(data.selected_citation_ids || []);
      // Build citation cards from returned context chunks.
      const items = list.map((c, i) => {
        const isSelected = selected.has(c.chunk_id);
        const long = (c.text || "").length > 220;
        return `
          <div class="citation ${isSelected ? "selected" : ""}">
            <div class="citation-head">
              <span class="citation-label">[${escapeHtml(c.label)}]</span>
              <span class="citation-source">${escapeHtml(c.source || "document")}</span>
              ${isSelected ? '<span class="citation-badge">used</span>' : ""}
            </div>
            <div class="citation-text ${long ? "" : "expanded"}" id="cite-${i}">${escapeHtml(c.text || "")}</div>
            ${long ? `<button class="citation-toggle" onclick="document.getElementById('cite-${i}').classList.toggle('expanded')">Show more / less</button>` : ""}
          </div>`;
      }).join("");
      return `
        <div style="margin-top: 24px; padding-top: 20px; border-top: 1px solid var(--line-soft);">
          <div class="citations-title">Evidence (${list.length})</div>
          ${items}
        </div>`;
    }

    function renderResult(data) {
      // Replace the empty card with the result.
      const answerText = data.answer || "No answer produced.";
      $("answerContainer").innerHTML = `
        <div class="answer-text">${escapeHtml(answerText)}</div>
        ${statusLine(data)}
        ${settingsLine(data)}
        ${citations(data)}
      `;
    }

    async function submit(mode) {
      // Send the current question to the local API.
      setBusy(true);
      try {
        const res = await fetch("/api/ask", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            question: $("question").value,
            mode,
            model: $("model").value,
            prompt_id: $("promptId").value,
            top_k: Number($("topK").value),
            latency_target_ms: Number($("latencyTarget").value),
            window_size: Number($("windowSize").value),
            overlap: Number($("overlap").value),
            max_context_tokens: Number($("contextTokens").value),
            use_generator: useGenerator.checked,
            self_correct: $("selfCorrect").checked
          })
        });
        // Error responses use the same JSON shape.
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Request failed");
        renderResult(data);
      } catch (err) {
        $("answerContainer").innerHTML =
          `<div class="status-line error">
            <span class="status-icon">▲</span>
            Error: ${escapeHtml(err.message)}
          </div>`;
      } finally {
        setBusy(false);
      }
    }

    askBtn.addEventListener("click", () => submit("answer"));
    summaryBtn.addEventListener("click", () => submit("summary"));
    $("question").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit("answer");
    });
    useGenerator.addEventListener("change", () => {
      $("model").disabled = !useGenerator.checked;
      $("selfCorrect").disabled = !useGenerator.checked;
    });
    refreshStatus();
  </script>
</body>
</html>
"""


def json_response(handler, payload, status=200):
    # Send all API replies as UTF-8 JSON.
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def squad_documents():
    """Load SQuAD paragraphs as named documents."""
    # Return an empty list when the dataset is missing.
    if not os.path.exists(SQUAD_DEV_PATH):
        return []
    with open(SQUAD_DEV_PATH, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    docs = []
    for article in data.get("data", []):
        title = article.get("title", "SQuAD")
        for index, paragraph in enumerate(article.get("paragraphs", [])):
            # Each paragraph becomes a searchable document.
            context = paragraph.get("context", "")
            if context:
                docs.append((f"{title} #{index}", context))
    return docs


def selected_int(payload, name, default, allowed):
    """Read a numeric setting from an allowed set."""
    value = int(payload.get(name, default))
    if value not in allowed:
        raise ValueError(f"invalid_{name}")
    return value


class ResearchHTTPServer(ThreadingHTTPServer):
    """HTTP server with reusable document indexes."""

    def __init__(self, address, handler, documents):
        super().__init__(address, handler)
        self.documents = documents
        self.indexes = {}
        self.index_lock = threading.Lock()

    def get_pipeline(self, window_size, overlap, top_k, model=""):
        # Chunking settings decide which index can be reused.
        key = (window_size, overlap)
        with self.index_lock:
            indexed = self.indexes.get(key)
            if indexed is None:
                indexed = LocalRAGPipeline(
                    window_size=window_size,
                    overlap=overlap,
                    top_k=top_k,
                )
                indexed.load_texts(self.documents)
                self.indexes[key] = indexed

        # Request state stays separate from the shared index.
        pipeline = LocalRAGPipeline(
            window_size=window_size,
            overlap=overlap,
            top_k=top_k,
            generator=LocalLLM(model=model or None),
        )
        pipeline.chunks = indexed.chunks
        pipeline.vectors = indexed.vectors
        pipeline.df = indexed.df
        pipeline.N = indexed.N
        return pipeline


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalQA/1.0"

    def log_message(self, fmt, *args):
        # Keep the standard server log format.
        print("%s - %s" % (self.address_string(), fmt % args))

    def do_GET(self):
        # Serve the page and status endpoint.
        if self.path == "/" or self.path.startswith("/?"):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/status":
            local_model = LocalLLM().status()
            pipeline = self.server.get_pipeline(100, 20, 3)
            json_response(self, {
                "local_model": local_model,
                "squad_available": bool(pipeline.chunks),
                "indexed_chunks": len(pipeline.chunks),
            })
            return

        json_response(self, {"error": "not_found"}, status=404)

    def do_POST(self):
        # Only the question endpoint accepts POST.
        if self.path != "/api/ask":
            json_response(self, {"error": "not_found"}, status=404)
            return

        try:
            # Read only the declared request body.
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            json_response(self, {"error": "invalid_json"}, status=400)
            return

        try:
            mode = payload.get("mode", "answer")
            if mode not in {"answer", "summary"}:
                raise ValueError("invalid_mode")
            prompt_id = payload.get("prompt_id", DEFAULT_PROMPT_ID)
            if prompt_id not in PROMPT_IDS:
                raise ValueError("invalid_prompt_id")
            top_k = selected_int(payload, "top_k", 3, TOP_K_VALUES)
            window_size = selected_int(payload, "window_size", 100, WINDOW_VALUES)
            overlap = selected_int(payload, "overlap", 20, OVERLAP_VALUES)
            latency_target_ms = selected_int(
                payload, "latency_target_ms", 5000, LATENCY_VALUES
            )
            max_context_tokens = selected_int(
                payload, "max_context_tokens", 4000, CONTEXT_VALUES
            )
        except (TypeError, ValueError) as exc:
            json_response(self, {"error": str(exc)}, status=400)
            return

        model = str(payload.get("model", ""))[:100]
        pipeline = self.server.get_pipeline(
            window_size, overlap, top_k, model=model
        )
        if not pipeline.chunks:
            json_response(self, {
                "error": "SQuAD dev set not found at data/dev-v2.0.json"
            }, status=400)
            return

        try:
            # Run with the settings selected in the UI.
            result = pipeline.answer(
                query=payload.get("question", ""),
                mode=mode,
                prompt_id=prompt_id,
                use_generator=bool(payload.get("use_generator", True)),
                self_correct=bool(payload.get("self_correct", False)),
                latency_target_ms=latency_target_ms,
                max_context_tokens=max_context_tokens,
            )
        except Exception as exc:
            json_response(self, {"error": str(exc)}, status=500)
            return

        json_response(self, result)


def main():
    # Start the local threaded server.
    parser = argparse.ArgumentParser(description="Run the local QA web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    documents = squad_documents()
    server = ResearchHTTPServer((args.host, args.port), Handler, documents)
    pipeline = server.get_pipeline(100, 20, 3)
    print(f"Sustainable Extractive QA running at http://{args.host}:{args.port}")
    print(f"Indexed chunks: {len(pipeline.chunks)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
