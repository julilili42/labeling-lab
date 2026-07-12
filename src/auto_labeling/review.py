from __future__ import annotations

from pathlib import Path

from .jsonl import read_jsonl, write_jsonl
from .teacher import RESEARCH_TERMS


def review_priority(label: dict[str, object]) -> tuple[int, float, str]:
    text = " ".join(str(label.get(key) or "") for key in ("url", "title", "snippet")).lower()
    is_researchish = any(term in text for term in RESEARCH_TERMS)
    label_value = str(label.get("label") or "")
    confidence = float(label.get("confidence") or 0.0)
    if label_value == "gray":
        return (0, confidence, str(label.get("url") or ""))
    if confidence < 0.80:
        return (1, confidence, str(label.get("url") or ""))
    if is_researchish and label_value == "positive":
        return (2, confidence, str(label.get("url") or ""))
    return (3, -confidence, str(label.get("url") or ""))


def make_review_batch(
    labels_path: Path,
    out_dir: Path,
    *,
    batch_size: int = 500,
    reviewed_path: Path | None = None,
) -> Path:
    reviewed_urls = set()
    if reviewed_path is not None:
        reviewed_urls = {str(row.get("normalized_url") or row.get("url") or "") for row in read_jsonl(reviewed_path)}
    labels = [
        row
        for row in read_jsonl(labels_path)
        if str(row.get("normalized_url") or row.get("url") or "") not in reviewed_urls
    ]
    labels.sort(key=review_priority)
    batch = labels[:batch_size]
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("batch-*.jsonl"))
    number = len(existing) + 1
    path = out_dir / f"batch-{number:04d}.jsonl"
    write_jsonl(path, batch)
    return path
