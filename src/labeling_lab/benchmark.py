"""Frozen-snapshot comparison helpers; benchmark files never join training data."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import joblib
from .link_features import LinkVerdictInput, make_text as make_link_text


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _space(value: object) -> str:
    return " ".join(str(value or "").split())


def _display_url(url: str) -> str:
    parsed = urlsplit(url)
    return f"{(parsed.hostname or '').removeprefix('www.')}/{parsed.path.strip('/')}".strip("/")


def page_text(row: dict[str, object]) -> str:
    url = _space(row.get("url"))
    return "\n".join((
        f"title: {_space(row.get('title'))}",
        f"url: {url}",
        f"display_url: {_space(row.get('display_url')) or _display_url(url)}",
        f"snippet: {_space(row.get('model_snippet') or row.get('snippet'))}",
        f"text: {_space(row.get('text'))[:3000]}",
    ))


def link_text(row: dict[str, object]) -> str:
    return make_link_text(LinkVerdictInput(
        anchor=str(row.get("anchor") or ""), target_url=str(row.get("target_url") or ""),
        parent_url=str(row.get("parent_url") or ""), parent_host=str(row.get("parent_host") or ""),
        parent_depth=row.get("parent_depth"), parent_pageverdict_score=row.get("parent_pageverdict_score"),
        parent_pageverdict_decision=str(row.get("parent_pageverdict_decision") or ""),
        parent_relevance=row.get("parent_relevance"), target_host=str(row.get("target_host") or ""),
        target_depth=row.get("target_depth"), raw_score=row.get("raw_score"),
    ))


def score(path: Path, rows: list[dict[str, object]], *, kind: str) -> list[float]:
    if not rows:
        return []
    bundle = joblib.load(path)
    model = bundle["model"]
    texts = [page_text(row) if kind == "page" else link_text(row) for row in rows]
    positive = list(model.classes_).index("positive")
    return [float(value) for value in model.predict_proba(texts)[:, positive]]


def threshold(path: Path) -> float:
    return float(joblib.load(path).get("positive_threshold", .5))


def item_id(row: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def _training_hashes(paths: list[Path]) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        if path.is_file():
            hashes |= {str(row.get("text_hash")) for row in read_jsonl(path) if row.get("text_hash")}
    return hashes


def _snapshot_hash(row: dict[str, object]) -> str:
    return hashlib.sha256(_space(row.get("text")).lower().encode()).hexdigest()


def create_review_queue(
    snapshot: Path, output: Path, *, exclude: list[Path], per_stratum: int = 10
) -> list[dict[str, object]]:
    rows = read_jsonl(snapshot)
    excluded = _training_hashes(exclude)
    pages = [row for row in rows if row.get("kind") == "page" and _snapshot_hash(row) not in excluded]
    links = [row for row in rows if row.get("kind") == "link"]
    accepted = [row for row in pages if row.get("outcome") == "accepted"]
    borderline = sorted(
        (row for row in pages if row.get("outcome") == "rejected" and row.get("score") is not None),
        key=lambda row: abs(float(row["score"]) - .6),
    )
    ranked = sorted((row for row in links if row.get("linkverdict_score") is not None), key=lambda row: float(row["linkverdict_score"]), reverse=True)
    strata = (("accepted_page", accepted), ("borderline_page", borderline), ("high_link", ranked), ("low_link", list(reversed(ranked))))
    queue: list[dict[str, object]] = []
    seen: set[str] = set()
    for stratum, candidates in strata:
        for row in candidates:
            identifier = item_id(row)
            if identifier in seen:
                continue
            seen.add(identifier)
            queue.append({"id": identifier, "kind": row["kind"], "stratum": stratum, "item": row})
            if sum(entry["stratum"] == stratum for entry in queue) == per_stratum:
                break
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in queue), encoding="utf-8")
    return queue


def gold_metrics(rows: list[dict[str, object]], scores: list[float], reviews: dict[str, str], *, kind: str, threshold: float) -> dict[str, object]:
    reviewed = [(row, score, reviews[item_id(row)]) for row, score in zip(rows, scores, strict=True) if item_id(row) in reviews]
    if not reviewed:
        return {"reviewed": 0}
    labels = [label for _, _, label in reviewed]
    predictions = [score >= threshold for _, score, _ in reviewed]
    positives = [label == "positive" for label in labels]
    tp = sum(actual and predicted for actual, predicted in zip(positives, predictions, strict=True))
    fp = sum(not actual and predicted for actual, predicted in zip(positives, predictions, strict=True))
    fn = sum(actual and not predicted for actual, predicted in zip(positives, predictions, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if kind == "link":
        ranked = sorted(reviewed, key=lambda entry: entry[1], reverse=True)
        k = min(10, len(ranked))
        return {"reviewed": len(reviewed), "precision_at_k": sum(label == "positive" for _, _, label in ranked[:k]) / k, "recall": recall, "k": k}
    return {"reviewed": len(reviewed), "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "false_accepts": fp, "false_rejects": fn}


def crawl_metrics(rows: list[dict[str, object]], reviews: dict[str, str]) -> dict[str, object]:
    pages = [row for row in rows if row.get("kind") == "page"]
    accepted = [row for row in pages if row.get("outcome") == "accepted"]
    rejected = [row for row in pages if row.get("outcome") == "rejected"]
    good = sum(reviews.get(item_id(row)) == "positive" for row in pages)
    return {"fetched_pages": len(pages), "index_accepts": len(accepted), "accept_rate": len(accepted) / len(pages) if pages else 0.0, "distinct_hosts": len({row.get('host') for row in accepted}), "rejection_reasons": dict(Counter(str(row.get("exclusion_reason")) for row in rejected)), "verified_good_per_100_fetches": 100 * good / len(pages) if pages else 0.0}


def comparison_report(snapshot: Path, baseline_page: Path, candidate_page: Path, baseline_link: Path, candidate_link: Path, reviews_path: Path) -> dict[str, object]:
    rows = read_jsonl(snapshot)
    reviews = {str(row["id"]): str(row["label"]) for row in read_jsonl(reviews_path) if row.get("label") in {"positive", "negative"}}
    pages = [row for row in rows if row.get("kind") == "page" and row.get("text")]
    links = [row for row in rows if row.get("kind") == "link" and row.get("linkverdict_score") is not None]
    return {"snapshot": str(snapshot), "baseline": {"page": str(baseline_page), "link": str(baseline_link)}, "candidate": {"page": str(candidate_page), "link": str(candidate_link)}, "page": {"baseline": gold_metrics(pages, score(baseline_page, pages, kind="page"), reviews, kind="page", threshold=threshold(baseline_page)), "candidate": gold_metrics(pages, score(candidate_page, pages, kind="page"), reviews, kind="page", threshold=threshold(candidate_page))}, "link": {"baseline": gold_metrics(links, score(baseline_link, links, kind="link"), reviews, kind="link", threshold=threshold(baseline_link)), "candidate": gold_metrics(links, score(candidate_link, links, kind="link"), reviews, kind="link", threshold=threshold(candidate_link))}, "crawl": crawl_metrics(rows, reviews)}
