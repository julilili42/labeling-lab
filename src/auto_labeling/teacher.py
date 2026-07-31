from __future__ import annotations

import json
import re
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

PROMPT_VERSION = "page-v4"
SYSTEM_PROMPT = (
    Path(__file__).parent / "prompts" / f"{PROMPT_VERSION}.txt"
).read_text(encoding="utf-8").rstrip("\n")

PAGE_PROMPT_HASH = stable_hash(SYSTEM_PROMPT)

GERMAN_TITLE_TERMS = (
    "angebote",
    "bewertungen",
    "buchung",
    "günstig",
    "nacht",
    "preise",
    "suchen",
    "unterkünfte",
    "verfügbarkeit",
)


def page_excerpt(snapshot: dict[str, object], *, max_chars: int = 2000) -> str:
    return str(snapshot.get("text") or "")[:max_chars]


def page_model_snippet(snapshot: dict[str, object], *, max_chars: int = 700) -> str:
    description = " ".join(str(snapshot.get("description") or "").split())
    h1 = " ".join(str(snapshot.get("h1") or "").split())
    parts = (description, h1 if h1 not in description else "")
    return " ".join(part for part in parts if part)[:max_chars]


def build_prompt(snapshot: dict[str, object], *, max_chars: int = 2000) -> str:
    h2 = ", ".join(str(item) for item in snapshot.get("h2") or [])
    return f"""Label this page.

URL:
{snapshot.get("url") or ""}

HTML language declaration:
{snapshot.get("html_lang") or "unknown"}

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


def _declares_german(raw: dict[str, object]) -> bool:
    if str(raw.get("html_lang") or "").lower().startswith("de"):
        return True
    heading = " ".join(
        str(raw.get(field) or "")
        for field in ("title", "target_title", "description", "h1")
    )
    return sum(bool(re.search(rf"\b{term}\b", heading, flags=re.I)) for term in GERMAN_TITLE_TERMS) >= 2


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

    if _declares_german(raw):
        english = False
        reason = "German destination evidence in HTML language or page heading."

    if error:
        label = "gray"
    elif confidence < 0.70:
        label = "gray"
        reason = reason or "Low teacher confidence."
    elif not english or not tuebingen:
        label = "negative"
    elif research_only:
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
    prompt: str | None = None,
    system: str = SYSTEM_PROMPT,
    endpoint: str = "http://127.0.0.1:11434/api/generate",
    timeout: int = 120,
) -> tuple[dict[str, object], str, dict[str, object]]:
    payload = json.dumps(
        {
            "model": model,
            "system": system,
            "prompt": prompt or build_prompt(snapshot, max_chars=max_chars),
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
    prompt_hash: str,
    max_chars: int,
) -> str:
    return stable_hash(
        {
            "teacher": teacher,
            "model": model,
            "prompt_version": prompt_version,
            "prompt_hash": prompt_hash,
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
    endpoint: str = "http://127.0.0.1:11434/api/generate",
) -> dict[str, object]:
    key = cache_key(
        snapshot,
        teacher=teacher,
        model=model,
        prompt_version=prompt_version,
        prompt_hash=PAGE_PROMPT_HASH,
        max_chars=max_chars,
    )
    cache_path = cache_dir / f"{key}.json"
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    if teacher == "mock":
        final = postprocess_label({**snapshot, **mock_label(snapshot)})
        raw_response = ""
        metrics = {}
    elif teacher == "ollama":
        final, raw_response, metrics = ollama_label(
            snapshot, model=model, max_chars=max_chars, endpoint=endpoint
        )
        final = postprocess_label({**snapshot, **final})
    else:
        raise ValueError(f"Unknown teacher: {teacher}")

    rating = 5 if final["label"] == "positive" else 1 if final["label"] == "negative" else 3
    record = {
        "snapshot_id": snapshot.get("id") or "",
        "url": snapshot.get("url") or "",
        "normalized_url": snapshot.get("normalized_url") or "",
        "host": snapshot.get("host") or "",
        "title": snapshot.get("title") or snapshot.get("serp_title") or "",
        "snippet": page_model_snippet(snapshot),
        "text_hash": snapshot.get("text_hash") or "",
        "teacher": teacher,
        "model": model,
        "prompt_version": prompt_version,
        "prompt_sha256": PAGE_PROMPT_HASH,
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
