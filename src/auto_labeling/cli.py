from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .fetcher import fetch_snapshot
from .jsonl import append_jsonl, read_jsonl, utc_now, write_jsonl
from .queries import read_queries
from .review import make_review_batch
from .serper import SERPER_PAGE_SIZE, dedupe_serp_rows, serper_page
from .teacher import (
    LINK_PROMPT_HASH,
    LINK_PROMPT_VERSION,
    PAGE_PROMPT_HASH,
    PROMPT_VERSION,
    label_link_snapshot,
    label_snapshot,
)
from .urls import normalize_url

ROOT = Path.cwd()
DATA = ROOT / "data"


def env_value(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    env_path = ROOT / ".env"
    if not env_path.exists():
        return ""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        if key.strip() == name:
            return raw_value.strip().strip('"').strip("'")
    return ""


def query_progress(rows: list[dict[str, object]]) -> dict[str, int]:
    progress: dict[str, int] = {}
    for row in rows:
        query = str(row.get("query") or "")
        if query:
            progress[query] = max(progress.get(query, 0), int(row.get("rank") or 0))
    return progress


def _resume_rows(
    path: Path,
    *,
    resume: bool,
    refresh: bool,
    prompt_version: str,
    prompt_hash: str,
    teacher: str,
    model: str,
) -> list[dict[str, object]]:
    if resume and refresh:
        raise SystemExit("--resume and --refresh cannot be combined; use a new output file")
    if not resume or not path.exists():
        return []
    rows = list(read_jsonl(path))
    expected = (prompt_version, prompt_hash, teacher, model)
    actual = {
        (
            str(row.get("prompt_version") or ""),
            str(row.get("prompt_sha256") or ""),
            str(row.get("teacher") or ""),
            str(row.get("model") or ""),
        )
        for row in rows
    }
    if actual - {expected}:
        raise SystemExit(
            f"{path} contains labels from another prompt or model; use a new output file"
        )
    return rows


def cmd_init(args: argparse.Namespace) -> None:
    (DATA / "review_batches").mkdir(parents=True, exist_ok=True)
    (DATA / "models").mkdir(parents=True, exist_ok=True)
    (DATA / "cache").mkdir(parents=True, exist_ok=True)
    queries = DATA / "queries.txt"
    if not queries.exists():
        queries.write_text(SAMPLE_QUERIES, encoding="utf-8")
    print(f"Initialized {ROOT}")


def cmd_search(args: argparse.Namespace) -> None:
    specs = read_queries(args.queries, limit=args.limit_queries)
    existing = list(read_jsonl(args.out)) if args.resume and args.out.exists() else []
    progress = query_progress(existing)
    calls = sum(max(0, (spec.results - progress.get(spec.query, 0) + 9) // 10) for spec in specs)
    print(f"Queries: {len(specs)}; planned Serper calls: {calls}")
    if args.dry_run:
        return
    api_key = env_value("SERPER_API_KEY")
    if not api_key:
        raise SystemExit("SERPER_API_KEY is not set")
    if not args.resume:
        write_jsonl(args.out, [])
    written = len(existing)
    for spec in specs:
        rank = progress.get(spec.query, 0) if args.resume else 0
        page = rank // SERPER_PAGE_SIZE + 1
        while rank < spec.results:
            remaining = spec.results - rank
            items = serper_page(
                spec.query,
                page=page,
                num=min(SERPER_PAGE_SIZE, remaining),
                api_key=api_key,
                timeout=args.timeout,
            )
            for item in items:
                url = str(item.get("link") or "")
                if not url:
                    continue
                rank += 1
                append_jsonl(
                    args.out,
                    {
                        "query": spec.query,
                        "rank": rank,
                        "title": str(item.get("title") or ""),
                        "url": url,
                        "normalized_url": normalize_url(url),
                        "display_url": str(item.get("displayLink") or item.get("displayedLink") or ""),
                        "snippet": str(item.get("snippet") or ""),
                        "source": "serper",
                        "fetched_at": utc_now(),
                    },
                )
                written += 1
                if rank >= spec.results:
                    break
            page += 1
    unique = len(dedupe_serp_rows(list(read_jsonl(args.out))))
    print(f"Wrote {written} SERP rows ({unique} unique URLs) to {args.out}")


def cmd_fetch(args: argparse.Namespace) -> None:
    rows = dedupe_serp_rows(list(read_jsonl(args.input)))
    if args.resume and args.out.exists():
        existing = list(read_jsonl(args.out))
        seen = {str(row.get("normalized_url") or "") for row in existing}
    else:
        existing, seen = [], set()
    to_fetch = [row for row in rows if str(row.get("normalized_url") or normalize_url(str(row.get("url") or ""))) not in seen]
    to_fetch = to_fetch[: args.limit]
    print(f"Fetching {len(to_fetch)} pages")
    if args.dry_run:
        return
    if not args.resume:
        write_jsonl(args.out, [])
    snapshots = existing[:]
    # ponytail: no per-host politeness cap; add per-host semaphores if a site rate-limits us
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for snapshot in pool.map(lambda row: fetch_snapshot(row, timeout=args.timeout), to_fetch):
            snapshots.append(snapshot)
            append_jsonl(args.out, snapshot)
    ok = sum(1 for row in snapshots if not row.get("fetch_error"))
    print(f"Wrote {len(snapshots)} snapshots ({ok} fetchable HTML) to {args.out}")


def cmd_label(args: argparse.Namespace) -> None:
    snapshots = [row for row in read_jsonl(args.input) if not row.get("fetch_error") and row.get("text")]
    existing = _resume_rows(
        args.out,
        resume=args.resume,
        refresh=args.refresh,
        prompt_version=PROMPT_VERSION,
        prompt_hash=PAGE_PROMPT_HASH,
        teacher=args.teacher,
        model=args.model,
    )
    seen = {str(row.get("text_hash") or "") for row in existing}
    candidates = [row for row in snapshots if str(row.get("text_hash") or "") not in seen][: args.limit]
    print(f"Labeling {len(candidates)} pages with {args.teacher}:{args.model}")
    if args.dry_run:
        return
    if not args.resume:
        write_jsonl(args.out, [])
    labels = existing[:]
    # ponytail: needs OLLAMA_NUM_PARALLEL >= workers on the server, else requests just queue
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        call = lambda snapshot: label_snapshot(
            snapshot,
            teacher=args.teacher,
            model=args.model,
            cache_dir=args.cache_dir,
            max_chars=args.max_input_chars,
            refresh=args.refresh,
            endpoint=args.endpoint,
        )
        for label in pool.map(call, candidates):
            labels.append(label)
            append_jsonl(args.out, label)
    print(f"Wrote {len(labels)} labels to {args.out}")


def cmd_label_links(args: argparse.Namespace) -> None:
    snapshots = {
        normalize_url(str(row.get("normalized_url") or row.get("url") or "")): row
        for row in read_jsonl(args.snapshots)
        if not row.get("fetch_error") and row.get("text")
    }
    existing = _resume_rows(
        args.out,
        resume=args.resume,
        refresh=args.refresh,
        prompt_version=LINK_PROMPT_VERSION,
        prompt_hash=LINK_PROMPT_HASH,
        teacher=args.teacher,
        model=args.model,
    )
    seen = {
        (
            normalize_url(str(row.get("parent_url") or "")),
            normalize_url(str(row.get("target_url") or row.get("url") or "")),
            " ".join(str(row.get("anchor") or "").split()),
        )
        for row in existing
    }
    pairs = []
    for link in read_jsonl(args.links):
        target = normalize_url(str(link.get("target_url") or link.get("url") or ""))
        snapshot = snapshots.get(target)
        key = (
            normalize_url(str(link.get("parent_url") or "")),
            target,
            " ".join(str(link.get("anchor") or "").split()),
        )
        if target and key not in seen and snapshot is not None:
            seen.add(key)
            pairs.append((link, snapshot))
    pairs = pairs[: args.limit]
    print(f"Labeling {len(pairs)} links with {args.teacher}:{args.model}")
    if args.dry_run:
        return
    if not args.resume:
        write_jsonl(args.out, [])
    labels = existing[:]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        call = lambda pair: label_link_snapshot(
            pair[0], pair[1], teacher=args.teacher, model=args.model,
            cache_dir=args.cache_dir, max_chars=args.max_input_chars, refresh=args.refresh,
            endpoint=args.endpoint,
        )
        for label in pool.map(call, pairs):
            labels.append(label)
            append_jsonl(args.out, label)
    print(f"Wrote {len(labels)} link labels to {args.out}")


def cmd_make_review_batch(args: argparse.Namespace) -> None:
    path = make_review_batch(args.labels, args.out_dir, batch_size=args.batch_size, reviewed_path=args.reviewed)
    print(f"Wrote review batch {path}")


def cmd_apply_reviews(args: argparse.Namespace) -> None:
    reviewed = {
        str(row.get("text_hash") or ""): row
        for row in read_jsonl(args.reviewed)
        if row.get("text_hash") and str(row.get("label") or "") in {"positive", "negative", "gray"}
    }
    rows = list(read_jsonl(args.labels))
    snapshots = {str(row.get("text_hash") or ""): row for row in read_jsonl(args.snapshots)}
    holdout_hashes = {str(row.get("text_hash") or "") for row in read_jsonl(args.holdout)}
    known_hashes = {str(row.get("text_hash") or "") for row in rows}
    added = 0
    for row in read_jsonl(args.teacher):
        text_hash = str(row.get("text_hash") or "")
        snapshot = snapshots.get(text_hash)
        if not text_hash or text_hash in known_hashes or text_hash in holdout_hashes or not snapshot:
            continue
        rows.append({**row, "text": snapshot.get("text") or ""})
        known_hashes.add(text_hash)
        added += 1
    changed = 0
    for row in rows:
        review = reviewed.get(str(row.get("text_hash") or ""))
        if review and row.get("label") != review.get("label"):
            row["label"] = review["label"]
            row["rating"] = review.get("rating")
            row["reviewed_at"] = review.get("reviewed_at")
            row["review_source"] = review.get("review_source")
            changed += 1
    write_jsonl(args.labels, rows)
    print(f"Added {added} teacher labels and applied {changed} reviews to {args.labels}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auto-label")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.set_defaults(func=cmd_init)

    search = sub.add_parser("search")
    search.add_argument("--queries", type=Path, default=DATA / "queries.txt")
    search.add_argument("--out", type=Path, default=DATA / "serp_results.jsonl")
    search.add_argument("--limit-queries", type=int, default=100)
    search.add_argument("--timeout", type=int, default=20)
    search.add_argument("--dry-run", action="store_true")
    search.add_argument("--resume", action="store_true")
    search.set_defaults(func=cmd_search)

    fetch = sub.add_parser("fetch")
    fetch.add_argument("--in", dest="input", type=Path, default=DATA / "serp_results.jsonl")
    fetch.add_argument("--out", type=Path, default=DATA / "page_snapshots.jsonl")
    fetch.add_argument("--limit", type=int, default=100)
    fetch.add_argument("--timeout", type=int, default=15)
    fetch.add_argument("--workers", type=int, default=8)
    fetch.add_argument("--dry-run", action="store_true")
    fetch.add_argument("--resume", action="store_true")
    fetch.set_defaults(func=cmd_fetch)

    label = sub.add_parser("label")
    label.add_argument("--in", dest="input", type=Path, default=DATA / "page_snapshots.jsonl")
    label.add_argument("--out", type=Path, default=DATA / "teacher_labels.raw.jsonl")
    label.add_argument("--teacher", choices=["mock", "ollama"], default="ollama")
    label.add_argument("--model", default="qwen2.5:7b")
    label.add_argument("--cache-dir", type=Path, default=DATA / "cache")
    label.add_argument("--limit", type=int, default=100)
    label.add_argument("--max-input-chars", type=int, default=2000)
    label.add_argument("--workers", type=int, default=4)
    label.add_argument("--dry-run", action="store_true")
    label.add_argument("--resume", action="store_true")
    label.add_argument("--refresh", action="store_true")
    label.add_argument("--endpoint", default="http://127.0.0.1:11434/api/generate")
    label.set_defaults(func=cmd_label)

    label_links = sub.add_parser("label-links")
    label_links.add_argument("--links", type=Path, default=DATA / "link_exploration_sample" / "candidates.jsonl")
    label_links.add_argument("--snapshots", type=Path, default=DATA / "link_exploration_sample" / "snapshots.jsonl")
    label_links.add_argument("--out", type=Path, default=DATA / "link_exploration_sample" / "teacher_labels.jsonl")
    label_links.add_argument("--teacher", choices=["mock", "ollama"], default="ollama")
    label_links.add_argument("--model", default="qwen2.5:7b")
    label_links.add_argument("--cache-dir", type=Path, default=DATA / "cache")
    label_links.add_argument("--limit", type=int, default=100)
    label_links.add_argument("--max-input-chars", type=int, default=2000)
    label_links.add_argument("--workers", type=int, default=4)
    label_links.add_argument("--dry-run", action="store_true")
    label_links.add_argument("--resume", action="store_true")
    label_links.add_argument("--refresh", action="store_true")
    label_links.add_argument("--endpoint", default="http://127.0.0.1:11434/api/generate")
    label_links.set_defaults(func=cmd_label_links)

    batch = sub.add_parser("make-review-batch")
    batch.add_argument("--labels", type=Path, default=DATA / "teacher_labels.raw.jsonl")
    batch.add_argument("--out-dir", type=Path, default=DATA / "review_batches")
    batch.add_argument("--reviewed", type=Path, default=DATA / "labels.reviewed.jsonl")
    batch.add_argument("--batch-size", type=int, default=500)
    batch.set_defaults(func=cmd_make_review_batch)

    apply_reviews = sub.add_parser("apply-reviews")
    apply_reviews.add_argument("--labels", type=Path, default=DATA / "labels.final.jsonl")
    apply_reviews.add_argument("--reviewed", type=Path, default=DATA / "labels.reviewed.jsonl")
    apply_reviews.add_argument("--teacher", type=Path, default=DATA / "teacher_labels.raw.jsonl")
    apply_reviews.add_argument("--snapshots", type=Path, default=DATA / "page_snapshots.jsonl")
    apply_reviews.add_argument("--holdout", type=Path, default=DATA / "eval_holdout.jsonl")
    apply_reviews.set_defaults(func=cmd_apply_reviews)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


SAMPLE_QUERIES = """# Core positives
Tuebingen tourism | 20
Tuebingen attractions | 20
Tuebingen events English | 20
Tuebingen student life | 20
Tuebingen accommodation | 20
Tuebingen transport | 20
Tuebingen restaurants English | 20
Tuebingen museums English | 20
University of Tuebingen overview | 20
University of Tuebingen admissions | 20
Tuebingen hospital English | 20
Tuebingen jobs English | 20

# Research drift traps
Tuebingen research group | 20
Tuebingen publications | 20
Tuebingen lab | 20
Tuebingen clinical trial | 20
Tuebingen AI research institute | 20
University of Tuebingen research English | 20
Max Planck Tuebingen research | 20
Cyber Valley Tuebingen research | 20

# Non-English and nearby negatives
Tuebingen Veranstaltungen | 20
Tuebingen Stadtverwaltung | 20
Reutlingen tourism English | 20
Heidelberg university research group | 20
Freiburg tourism English | 20
Germany university research group | 20
"""


if __name__ == "__main__":
    main()
