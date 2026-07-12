from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .jsonl import append_jsonl, read_jsonl, utc_now

HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Auto Labeling Review</title>
  <style>
    body { font: 15px/1.45 system-ui, sans-serif; margin: 0; color: #17202a; background: #f6f7f9; }
    main { max-width: 980px; margin: 0 auto; padding: 24px; }
    .panel { background: white; border: 1px solid #d8dde5; border-radius: 8px; padding: 18px; }
    .meta { color: #5f6b7a; font-size: 13px; }
    .url { word-break: break-all; }
    .snippet { white-space: pre-wrap; max-height: 260px; overflow: auto; background: #f0f2f5; padding: 12px; border-radius: 6px; }
    button { margin-right: 8px; padding: 9px 12px; border: 1px solid #b7c0cc; background: white; border-radius: 6px; cursor: pointer; }
    button.primary { background: #214e8a; color: white; border-color: #214e8a; }
    textarea { width: 100%; min-height: 70px; margin: 10px 0; }
  </style>
</head>
<body>
<main>
  <h1>Auto Labeling Review</h1>
  <div class="panel">
    <div id="progress" class="meta"></div>
    <h2 id="title"></h2>
    <p class="url"><a id="url" target="_blank" rel="noreferrer"></a></p>
    <p id="teacher" class="meta"></p>
    <p id="reason"></p>
    <pre id="snippet" class="snippet"></pre>
    <textarea id="notes" placeholder="Notes"></textarea>
    <div>
      <button onclick="save('positive')" class="primary">Positive</button>
      <button onclick="save('negative')">Negative</button>
      <button onclick="save('gray')">Gray</button>
      <button onclick="next()">Skip</button>
    </div>
  </div>
</main>
<script>
let items = []
let index = 0
async function load() {
  items = await (await fetch('/api/items')).json()
  show()
}
function show() {
  if (index >= items.length) {
    document.querySelector('.panel').innerHTML = '<h2>Done</h2>'
    return
  }
  const item = items[index]
  progress.textContent = `${index + 1} / ${items.length}`
  title.textContent = item.title || '(no title)'
  url.href = item.url
  url.textContent = item.url
  teacher.textContent = `teacher: ${item.teacher} / ${item.model} | label: ${item.label} | confidence: ${item.confidence}`
  reason.textContent = item.reason || ''
  snippet.textContent = item.snippet || ''
  notes.value = ''
}
function next() { index += 1; show() }
async function save(label) {
  const item = items[index]
  await fetch('/api/review', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({index, label, notes: notes.value, item})
  })
  next()
}
load()
</script>
</body>
</html>
"""


class ReviewHandler(BaseHTTPRequestHandler):
    items: list[dict[str, object]] = []
    output_path: Path

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", HTML.encode("utf-8"))
        elif path == "/api/items":
            self._send(200, "application/json", json.dumps(self.items, ensure_ascii=False).encode("utf-8"))
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/review":
            self._send(404, "text/plain", b"not found")
            return
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        item = dict(payload.get("item") or {})
        teacher_label = item.get("label")
        label = payload.get("label")
        rating = 5 if label == "positive" else 1 if label == "negative" else 3
        item.update(
            {
                "teacher_label": teacher_label,
                "label": label,
                "rating": rating,
                "review_notes": payload.get("notes") or "",
                "reviewed_at": utc_now(),
                "review_source": "review_ui",
            }
        )
        append_jsonl(self.output_path, item)
        self._send(200, "application/json", b"{\"status\":\"ok\"}")


def run_review_server(batch_path: Path, output_path: Path, *, host: str, port: int) -> None:
    ReviewHandler.items = list(read_jsonl(batch_path))
    ReviewHandler.output_path = output_path
    server = ThreadingHTTPServer((host, port), ReviewHandler)
    print(f"Review UI: http://{host}:{port}")
    server.serve_forever()
