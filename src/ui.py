"""
Minimal local web UI for Variant 3 QA.

Run:
    python -m src.ui --host 127.0.0.1 --port 8000
"""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .local_llm import LocalLLM
from .model_registry import CANDIDATE_MODELS
from .rag_pipeline import LocalRAGPipeline


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DOCS = [
    os.path.join(PROJECT_ROOT, "README.md"),
    os.path.join(PROJECT_ROOT, "data", "README.md"),
    os.path.join(PROJECT_ROOT, "results", "full_results.md"),
]
SQUAD_DEV_PATH = os.path.join(PROJECT_ROOT, "data", "dev-v2.0.json")


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Local QA</title>
  <style>
    :root {
      --bg: #f7f7f4;
      --ink: #20231f;
      --muted: #62675f;
      --line: #d8d9d2;
      --surface: #ffffff;
      --accent: #1f7a5a;
      --accent-dark: #155f46;
      --warn: #9b5c00;
      --code: #eef2ea;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header {
      min-height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.86);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    h1 {
      font-size: 18px;
      line-height: 1.2;
      margin: 0;
      font-weight: 700;
    }
    main {
      display: grid;
      grid-template-columns: minmax(320px, 0.9fr) minmax(360px, 1.1fr);
      gap: 18px;
      padding: 18px;
      max-width: 1280px;
      margin: 0 auto;
    }
    section {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
    }
    .pane {
      padding: 16px;
    }
    .stack {
      display: grid;
      gap: 12px;
    }
    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    textarea, input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-size: 14px;
      outline: none;
    }
    textarea:focus, input:focus, select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(31, 122, 90, 0.14);
    }
    textarea {
      min-height: 280px;
      resize: vertical;
      padding: 10px;
      line-height: 1.45;
    }
    input, select {
      min-height: 40px;
      padding: 0 10px;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .checkrow {
      display: flex;
      align-items: center;
      gap: 10px;
      min-height: 40px;
      color: var(--ink);
      font-size: 14px;
      text-transform: none;
      font-weight: 600;
    }
    .checkrow input {
      width: 18px;
      min-height: 18px;
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    button {
      min-height: 40px;
      border: 1px solid var(--accent);
      background: var(--accent);
      color: white;
      border-radius: 6px;
      padding: 0 14px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    button.secondary {
      background: #fff;
      color: var(--accent-dark);
    }
    button:disabled {
      cursor: wait;
      opacity: 0.68;
    }
    .status {
      display: flex;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--warn);
    }
    .dot.ok { background: var(--accent); }
    .answer {
      min-height: 220px;
      white-space: pre-wrap;
      line-height: 1.55;
      font-size: 15px;
    }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
    }
    .pill {
      background: var(--code);
      color: #314133;
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
      font-weight: 700;
    }
    .citations {
      display: grid;
      gap: 8px;
      margin-top: 14px;
    }
    .citation {
      border-top: 1px solid var(--line);
      padding-top: 10px;
      font-size: 13px;
      line-height: 1.45;
    }
    .citation.selected {
      border-left: 3px solid var(--accent);
      padding-left: 10px;
    }
    .citation b {
      display: inline-block;
      margin-right: 6px;
    }
    .empty {
      color: var(--muted);
    }
    @media (max-width: 820px) {
      main {
        grid-template-columns: 1fr;
        padding: 12px;
      }
      header {
        align-items: flex-start;
        flex-direction: column;
        gap: 6px;
        padding: 12px;
      }
      .row {
        grid-template-columns: 1fr;
      }
      textarea {
        min-height: 220px;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>Local QA</h1>
    <div class="status"><span id="statusDot" class="dot"></span><span id="statusText">Checking local model</span></div>
  </header>
  <main>
    <section class="pane">
      <div class="stack">
        <label>Corpus
          <select id="corpus">
            <option value="project">Project docs</option>
            <option value="squad">SQuAD dev set</option>
          </select>
        </label>
        <label>Document
          <textarea id="documentText" placeholder="Optional. Pasted text overrides the selected corpus."></textarea>
        </label>
        <label>Question
          <input id="question" value="How many questions are in the dev set?">
        </label>
        <div class="row">
          <label>Prompt
            <select id="prompt">
              <option value="rag_cited_answer">Cited answer</option>
              <option value="rag_strict">Strict answer</option>
              <option value="graph_answer">Graph answer</option>
            </select>
          </label>
          <label>Model
            <input id="model" placeholder="Auto">
          </label>
        </div>
        <div class="row">
          <label>Target ms
            <input id="target" type="number" value="1500" min="50" step="50">
          </label>
          <label>Context tokens
            <input id="contextTokens" type="number" value="900" min="120" step="50">
          </label>
        </div>
        <label class="checkrow"><input id="useGenerator" type="checkbox" checked>Use local generator when available</label>
        <label class="checkrow"><input id="selfCorrect" type="checkbox">Self-correct within latency budget</label>
        <div class="actions">
          <button id="askBtn">Ask</button>
          <button id="summaryBtn" class="secondary">Summarize</button>
        </div>
      </div>
    </section>
    <section class="pane">
      <div class="stack">
        <div id="answer" class="answer empty">Answer will appear here.</div>
        <div id="meta" class="meta"></div>
        <div id="citations" class="citations"></div>
      </div>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const askBtn = $("askBtn");
    const summaryBtn = $("summaryBtn");

    async function refreshStatus() {
      try {
        const res = await fetch("/api/status");
        const data = await res.json();
        $("statusDot").classList.toggle("ok", data.local_model.available);
        const model = data.local_model.selected_model || "retrieval-only";
        $("statusText").textContent = data.local_model.available ? `Local model: ${model}` : "Retrieval-only fallback";
      } catch {
        $("statusText").textContent = "Status unavailable";
      }
    }

    function payload(mode) {
      return {
        document: $("documentText").value,
        corpus: $("corpus").value,
        question: $("question").value,
        mode,
        prompt_id: $("prompt").value,
        model: $("model").value.trim(),
        use_generator: $("useGenerator").checked,
        self_correct: $("selfCorrect").checked,
        latency_target_ms: Number($("target").value || 1500),
        max_context_tokens: Number($("contextTokens").value || 900)
      };
    }

    function setBusy(busy) {
      askBtn.disabled = busy;
      summaryBtn.disabled = busy;
      if (busy) {
        $("answer").classList.remove("empty");
        $("answer").textContent = "Working...";
        $("meta").innerHTML = "";
        $("citations").innerHTML = "";
      }
    }

    function renderResult(data) {
      $("answer").classList.remove("empty");
      $("answer").textContent = data.answer || "No answer.";
      const fallback = data.generated ? "generated" : `fallback: ${data.fallback_reason || "retrieval_only"}`;
      const pills = [
        `${data.latency_ms} ms`,
        `retrieval ${data.retrieval_latency_ms} ms`,
        `generation ${data.generation_latency_ms} ms`,
        fallback,
        `prompt ${data.prompt_id}`,
        `context ${data.context_tokens}`,
        `output ${data.output_tokens}`
      ];
      $("meta").innerHTML = pills.map((p) => `<span class="pill">${p}</span>`).join("");
      const selected = new Set(data.selected_citation_ids || []);
      $("citations").innerHTML = (data.citations || []).map((c) => {
        const klass = selected.has(c.chunk_id) ? "citation selected" : "citation";
        return `<div class="${klass}"><b>[${c.label}]</b>${c.source}<br>${c.text}</div>`;
      }).join("");
    }

    async function submit(mode) {
      setBusy(true);
      try {
        const res = await fetch("/api/ask", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload(mode))
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Request failed");
        renderResult(data);
      } catch (err) {
        $("answer").textContent = err.message;
      } finally {
        setBusy(false);
      }
    }

    askBtn.addEventListener("click", () => submit("answer"));
    summaryBtn.addEventListener("click", () => submit("summary"));
    refreshStatus();
  </script>
</body>
</html>
"""


def json_response(handler, payload, status=200):
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def default_documents():
    docs = []
    for path in DEFAULT_DOCS:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                docs.append((os.path.relpath(path, PROJECT_ROOT), handle.read()))
    return docs


def squad_documents():
    """Load SQuAD dev paragraphs as named documents when the file exists."""
    if not os.path.exists(SQUAD_DEV_PATH):
        return []
    with open(SQUAD_DEV_PATH, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    docs = []
    for article in data.get("data", []):
        title = article.get("title", "SQuAD")
        for index, paragraph in enumerate(article.get("paragraphs", [])):
            context = paragraph.get("context", "")
            if context:
                docs.append((f"{title} #{index}", context))
    return docs


def documents_for_payload(payload):
    """Resolve pasted text or the selected corpus into named documents."""
    document = (payload.get("document") or "").strip()
    if document:
        return [("Pasted document", document)]
    if payload.get("corpus") == "squad":
        return squad_documents()
    return default_documents()


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalQA/1.0"

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))

    def do_GET(self):
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
            json_response(self, {
                "local_model": local_model,
                "candidate_models": CANDIDATE_MODELS,
                "default_documents": [name for name, _ in default_documents()],
                "squad_available": os.path.exists(SQUAD_DEV_PATH),
            })
            return

        json_response(self, {"error": "not_found"}, status=404)

    def do_POST(self):
        if self.path != "/api/ask":
            json_response(self, {"error": "not_found"}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            json_response(self, {"error": "invalid_json"}, status=400)
            return

        documents = documents_for_payload(payload)
        if not documents:
            json_response(self, {"error": "no_document"}, status=400)
            return

        generator = LocalLLM(model=(payload.get("model") or "").strip() or None)
        pipeline = LocalRAGPipeline(generator=generator)
        pipeline.load_texts(documents)

        try:
            result = pipeline.answer(
                query=payload.get("question", ""),
                mode=payload.get("mode", "answer"),
                prompt_id=payload.get("prompt_id", "rag_cited_answer"),
                use_generator=bool(payload.get("use_generator", True)),
                self_correct=bool(payload.get("self_correct", False)),
                latency_target_ms=int(payload.get("latency_target_ms", 1500)),
                max_context_tokens=int(payload.get("max_context_tokens", 900)),
                model=(payload.get("model") or "").strip() or None,
            )
        except Exception as exc:
            json_response(self, {"error": str(exc)}, status=500)
            return

        json_response(self, result)


def main():
    parser = argparse.ArgumentParser(description="Run the local QA web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Local QA UI running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
