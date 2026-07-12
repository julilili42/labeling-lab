from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from .jsonl import stable_hash, utc_now

RESEARCH_TERMS = {
    "research",
    "publication",
    "publications",
    "paper",
    "papers",
    "conference",
    "dataset",
    "laboratory",
    "lab",
    "project",
    "clinical trial",
    "study",
    "studies",
    "doi",
    "arxiv",
}

PROMPT_VERSION = "page-v1"
SYSTEM_PROMPT = """You label web pages for an English search engine about Tuebingen, Germany.

Return only valid JSON. Do not include markdown.

Definitions:
- positive: mainly English, related to Tuebingen, and useful for a general English Tuebingen search engine.
- negative: non-English, not Tuebingen-related, or narrow specialist content.
- gray: mixed, ambiguous, too little evidence, or low confidence.

Research-only pages are negative. This includes papers, publication lists, lab news, datasets, technical project pages, clinical trial details, and individual researcher profiles mainly useful to specialists.

University or research institution overview pages can be positive only if they are useful to general users, students, visitors, patients, applicants, or people looking for practical information."""


def page_excerpt(snapshot: dict[str, object], *, max_chars: int = 2000) -> str:
    return str(snapshot.get("text") or "")[:max_chars]


def build_prompt(snapshot: dict[str, object], *, max_chars: int = 2000) -> str:
    h2 = ", ".join(str(item) for item in snapshot.get("h2") or [])
    return f"""{SYSTEM_PROMPT}

Label this page.

URL:
{snapshot.get("url") or ""}

Title:
{snapshot.get("title") or snapshot.get("serp_title") or ""}

Description:
{snapshot.get("description") or ""}

H1:
{snapshot.get("h1") or ""}

H2:
{h2}

Visible text excerpt:
{page_excerpt(snapshot, max_chars=max_chars)}

Return JSON with exactly these fields:
{{
  "english": boolean,
  "tuebingen_related": boolean,
  "general_search_useful": boolean,
  "research_only": boolean,
  "label": "positive" | "negative" | "gray",
  "confidence": number,
  "reason": string
}}"""


def postprocess_label(raw: dict[str, object], *, error: str = "") -> dict[str, object]:
    label = str(raw.get("label") or "gray").lower()
    if label not in {"positive", "negative", "gray"}:
        label = "gray"
    confidence = float(raw.get("confidence") or 0.0)
    if 1.0 < confidence <= 10.0:
        confidence = confidence / 10.0
    confidence = min(1.0, max(0.0, confidence))
    english = bool(raw.get("english"))
    tuebingen = bool(raw.get("tuebingen_related"))
    useful = bool(raw.get("general_search_useful"))
    research_only = bool(raw.get("research_only"))
    reason = str(raw.get("reason") or error or "")

    if error:
        label = "gray"
    elif confidence < 0.70:
        label = "gray"
        reason = reason or "Low teacher confidence."
    elif not english or not tuebingen:
        label = "negative"
    elif research_only and not useful:
        label = "negative"
    elif label == "positive" and not useful:
        label = "gray"

    return {
        "english": english,
        "tuebingen_related": tuebingen,
        "general_search_useful": useful,
        "research_only": research_only,
        "label": label,
        "confidence": confidence,
        "reason": reason,
    }


def mock_label(snapshot: dict[str, object]) -> dict[str, object]:
    haystack = " ".join(
        str(snapshot.get(key) or "")
        for key in ("url", "title", "serp_title", "description", "h1", "text")
    ).lower()
    tuebingen = any(term in haystack for term in ("tuebingen", "tübingen", "tubingen"))
    research = any(term in haystack for term in RESEARCH_TERMS)
    german = any(term in haystack for term in (" der ", " die ", " und ", "stadtverwaltung", "bürger"))
    label = "positive" if tuebingen and not research and not german else "negative"
    return postprocess_label(
        {
            "english": not german,
            "tuebingen_related": tuebingen,
            "general_search_useful": label == "positive",
            "research_only": research,
            "label": label,
            "confidence": 0.80,
            "reason": "Mock label from simple keyword policy.",
        }
    )


def ollama_label(
    snapshot: dict[str, object],
    *,
    model: str,
    max_chars: int,
    endpoint: str = "http://127.0.0.1:11434/api/generate",
    timeout: int = 120,
) -> tuple[dict[str, object], str, dict[str, object]]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": build_prompt(snapshot, max_chars=max_chars),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
    ).encode("utf-8")
    request = Request(endpoint, data=payload, headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        return postprocess_label({}, error=f"ollama_failed: {exc}"), "", {
            "duration_ms": round((time.monotonic() - started) * 1000)
        }
    raw_response = str(data.get("response") or "")
    metrics = {
        "duration_ms": round((time.monotonic() - started) * 1000),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
        "total_duration": data.get("total_duration"),
    }
    try:
        raw_label = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return postprocess_label({}, error=f"invalid_json: {exc}"), raw_response, metrics
    return postprocess_label(raw_label), raw_response, metrics


def cache_key(
    snapshot: dict[str, object],
    *,
    teacher: str,
    model: str,
    prompt_version: str,
    max_chars: int,
) -> str:
    return stable_hash(
        {
            "teacher": teacher,
            "model": model,
            "prompt_version": prompt_version,
            "max_chars": max_chars,
            "text_hash": snapshot.get("text_hash") or snapshot.get("id"),
        }
    )


def label_snapshot(
    snapshot: dict[str, object],
    *,
    teacher: str,
    model: str,
    cache_dir: Path,
    prompt_version: str = PROMPT_VERSION,
    max_chars: int = 2000,
    refresh: bool = False,
) -> dict[str, object]:
    key = cache_key(
        snapshot,
        teacher=teacher,
        model=model,
        prompt_version=prompt_version,
        max_chars=max_chars,
    )
    cache_path = cache_dir / f"{key}.json"
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    if teacher == "mock":
        final = mock_label(snapshot)
        raw_response = ""
        metrics = {}
    elif teacher == "ollama":
        final, raw_response, metrics = ollama_label(snapshot, model=model, max_chars=max_chars)
    else:
        raise ValueError(f"Unknown teacher: {teacher}")

    rating = 5 if final["label"] == "positive" else 1 if final["label"] == "negative" else 3
    record = {
        "snapshot_id": snapshot.get("id") or "",
        "url": snapshot.get("url") or "",
        "normalized_url": snapshot.get("normalized_url") or "",
        "host": snapshot.get("host") or "",
        "title": snapshot.get("title") or snapshot.get("serp_title") or "",
        "snippet": page_excerpt(snapshot, max_chars=700),
        "text_hash": snapshot.get("text_hash") or "",
        "teacher": teacher,
        "model": model,
        "prompt_version": prompt_version,
        "max_input_chars": max_chars,
        "raw_response": raw_response,
        "teacher_metrics": metrics,
        "rating": rating,
        **final,
        "labeled_at": utc_now(),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return record
