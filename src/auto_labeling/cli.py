from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .fetcher import fetch_snapshot
from .jsonl import append_jsonl, read_jsonl, utc_now, write_jsonl
from .queries import read_queries
from .review import make_review_batch
from .review_server import run_review_server
from .serper import SERPER_PAGE_SIZE, dedupe_serp_rows, serper_page
from .teacher import label_snapshot
from .train import evaluate_model, train_model
from .urls import normalize_url
from labeling_lab.link_train import main as train_link_model

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
    text_hashes = {str(row.get("text_hash") or "") for row in existing if row.get("text_hash")}
    # ponytail: no per-host politeness cap; add per-host semaphores if a site rate-limits us
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for snapshot in pool.map(lambda row: fetch_snapshot(row, timeout=args.timeout), to_fetch):
            text_hash = str(snapshot.get("text_hash") or "")
            if text_hash and text_hash in text_hashes:
                continue
            if text_hash:
                text_hashes.add(text_hash)
            snapshots.append(snapshot)
            append_jsonl(args.out, snapshot)
    ok = sum(1 for row in snapshots if not row.get("fetch_error"))
    print(f"Wrote {len(snapshots)} snapshots ({ok} fetchable HTML) to {args.out}")


def cmd_label(args: argparse.Namespace) -> None:
    snapshots = [row for row in read_jsonl(args.input) if not row.get("fetch_error") and row.get("text")]
    if args.resume and args.out.exists():
        existing = list(read_jsonl(args.out))
        seen = {str(row.get("text_hash") or "") for row in existing}
    else:
        existing, seen = [], set()
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
        )
        for label in pool.map(call, candidates):
            labels.append(label)
            append_jsonl(args.out, label)
    print(f"Wrote {len(labels)} labels to {args.out}")


def cmd_make_review_batch(args: argparse.Namespace) -> None:
    path = make_review_batch(args.labels, args.out_dir, batch_size=args.batch_size, reviewed_path=args.reviewed)
    print(f"Wrote review batch {path}")


def cmd_review_server(args: argparse.Namespace) -> None:
    run_review_server(args.batch, args.out, host=args.host, port=args.port)


def cmd_train(args: argparse.Namespace) -> None:
    report = train_model(args.labels, args.out, metrics_path=args.metrics)
    print(f"Trained {args.out}")
    print(f"Examples: {report['examples']}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    report = evaluate_model(args.model, args.holdout, metrics_path=args.metrics)
    print(f"Evaluated {args.model}")
    print(f"Examples: {report['examples']}")


def cmd_train_link(args: argparse.Namespace) -> None:
    argv = ["--db", str(args.db), "--out", str(args.out)]
    if args.crawl_db:
        argv.extend(["--crawl-db", str(args.crawl_db)])
    train_link_model(argv)


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
    label.set_defaults(func=cmd_label)

    batch = sub.add_parser("make-review-batch")
    batch.add_argument("--labels", type=Path, default=DATA / "teacher_labels.raw.jsonl")
    batch.add_argument("--out-dir", type=Path, default=DATA / "review_batches")
    batch.add_argument("--reviewed", type=Path, default=DATA / "labels.reviewed.jsonl")
    batch.add_argument("--batch-size", type=int, default=500)
    batch.set_defaults(func=cmd_make_review_batch)

    review = sub.add_parser("review-server")
    review.add_argument("--batch", type=Path, required=True)
    review.add_argument("--out", type=Path, default=DATA / "labels.reviewed.jsonl")
    review.add_argument("--host", default="127.0.0.1")
    review.add_argument("--port", type=int, default=8020)
    review.set_defaults(func=cmd_review_server)

    train = sub.add_parser("train")
    train.add_argument("--labels", type=Path, default=DATA / "labels.reviewed.jsonl")
    train.add_argument("--out", type=Path, default=DATA / "models" / "page_verdict.joblib")
    train.add_argument("--metrics", type=Path, default=DATA / "models" / "metrics.json")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--model", type=Path, default=DATA / "models" / "page_verdict.joblib")
    evaluate.add_argument("--holdout", type=Path, default=DATA / "eval_holdout.jsonl")
    evaluate.add_argument("--metrics", type=Path, default=DATA / "models" / "holdout_metrics.json")
    evaluate.set_defaults(func=cmd_evaluate)

    train_link = sub.add_parser("train-link")
    train_link.add_argument("--db", type=Path, default=DATA / "labeling.sqlite")
    train_link.add_argument("--out", type=Path, default=DATA / "models")
    train_link.add_argument("--crawl-db", type=Path)
    train_link.set_defaults(func=cmd_train_link)

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
