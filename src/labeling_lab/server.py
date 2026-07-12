from __future__ import annotations

import argparse
import io
import json
import os
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from auto_labeling.jsonl import append_jsonl, utc_now


ROOT_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = ROOT_DIR / "static"
DEFAULT_DB_PATH = ROOT_DIR / "data" / "labeling.sqlite"

SERPER_ENDPOINT = "https://google.serper.dev/search"
SERPER_PROVIDER = "serper"
SERPER_COUNT = 10
SERPER_PAGES = 4
SERPER_TIMEOUT_SECONDS = 20
DEFAULT_CANDIDATE_SOURCES = ("crawler_pageverdict", "pageverdict_error")

LabelValue = Literal["positive", "negative", "skip"]

app = FastAPI(title="Labeling Lab")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class JobState:
    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.command = ""
        self.output: list[str] = []
        self.lock = threading.Lock()

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            return {"running": running, "command": self.command, "output": self.output[-80:]}


JOB = JobState()


class SerpResult(BaseModel):
    id: int
    query: str
    page_number: int
    rank: int
    title: str
    url: str
    display_url: str
    snippet: str
    source: str
    rating: int | None = None
    label: LabelValue | None = None
    notes: str = ""
    pageverdict_score: float | None = None
    pageverdict_label: str | None = None
    pageverdict_decision: str | None = None
    pageverdict_model: str | None = None
    crawler_source_table: str | None = None
    crawler_exclusion_reason: str | None = None
    crawler_status_code: int | None = None
    crawler_content_type: str | None = None
    crawler_depth: int | None = None
    crawler_relevance: float | None = None
    crawler_token_count: int | None = None
    created_at: str
    updated_at: str
    rated_at: str | None = None


class LinkResult(BaseModel):
    id: int
    parent_url: str
    parent_host: str
    parent_depth: int | None = None
    parent_pageverdict_score: float | None = None
    parent_pageverdict_label: str | None = None
    parent_pageverdict_decision: str | None = None
    parent_relevance: float | None = None
    anchor: str
    target_url: str
    target_host: str
    target_depth: int | None = None
    raw_score: float | None = None
    linkverdict_score: float | None = None
    linkverdict_label: str | None = None
    linkverdict_model: str | None = None
    should_enqueue: bool
    selected: bool
    rejection_reason: str | None = None
    target_status: str | None = None
    target_status_code: int | None = None
    target_content_type: str | None = None
    target_language: str | None = None
    target_relevance: float | None = None
    target_token_count: int | None = None
    target_pageverdict_score: float | None = None
    target_pageverdict_label: str | None = None
    target_pageverdict_decision: str | None = None
    target_exclusion_reason: str | None = None
    target_fetched_at: str | None = None
    source: str
    rating: int | None = None
    label: LabelValue | None = None
    notes: str = ""
    created_at: str
    updated_at: str
    rated_at: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)


class SearchResponse(BaseModel):
    query: str
    source: str
    pages: int
    count_per_page: int
    results: list[SerpResult]


class RatingRequest(BaseModel):
    result_id: int
    rating: int | None = Field(default=None, ge=1, le=5)
    notes: str = ""


class LinkRatingRequest(BaseModel):
    link_id: int
    rating: int | None = Field(default=None, ge=1, le=5)
    notes: str = ""


class CrawlerImportRequest(BaseModel):
    path: str = "data/pageverdict_candidates.jsonl"
    query: str = "crawler:pageverdict"
    limit: int = Field(default=200, ge=1, le=2000)
    unlabeled_only: bool = True


class LinkImportRequest(BaseModel):
    path: str = "data/link_candidates.jsonl"
    limit: int = Field(default=200, ge=1, le=2000)
    unlabeled_only: bool = True


class ActionResponse(BaseModel):
    status: str = "ok"


class CrawlerImportResponse(BaseModel):
    status: str = "ok"
    path: str
    rows_read: int
    stored: int
    results: list[SerpResult]


class LinkImportResponse(BaseModel):
    status: str = "ok"
    path: str
    rows_read: int
    stored: int
    results: list[LinkResult]


class ReviewImportRequest(BaseModel):
    path: str


class ReviewRatingRequest(BaseModel):
    item: dict[str, object]
    rating: int = Field(ge=1, le=5)
    notes: str = ""


class LabelStats(BaseModel):
    results: int
    rated: int
    labels: dict[str, int]
    ratings: dict[str, int]


class PipelineRequest(BaseModel):
    action: Literal["search", "fetch", "label", "train-page", "train-link"]
    limit: int | None = Field(default=None, ge=1, le=10000)
    workers: int | None = Field(default=None, ge=1, le=32)
    model: str | None = None
    crawl_db: str | None = None


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as file:
        return sum(bool(line.strip()) for line in file)


def pipeline_status() -> dict[str, object]:
    return {
        "job": JOB.snapshot(),
        "data": {
            "serp_rows": _line_count(ROOT_DIR / "data" / "serp_results.jsonl"),
            "snapshots": _line_count(ROOT_DIR / "data" / "page_snapshots.jsonl"),
            "teacher_labels": _line_count(ROOT_DIR / "data" / "teacher_labels.raw.jsonl"),
            "reviewed_labels": _line_count(ROOT_DIR / "data" / "labels.reviewed.jsonl"),
            "page_model": (ROOT_DIR / "data" / "models" / "page_verdict.joblib").exists(),
            "link_model": (ROOT_DIR / "data" / "models" / "link_verdict.joblib").exists(),
        },
    }


def _read_job_output(process: subprocess.Popen[str]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        with JOB.lock:
            JOB.output.append(line.rstrip())
    process.wait()


def _pipeline_command(payload: PipelineRequest) -> list[str]:
    command = [sys.executable, "-m", "auto_labeling.cli"]
    match payload.action:
        case "search":
            command += ["search", "--resume", "--limit-queries", str(payload.limit or 300)]
        case "fetch":
            command += ["fetch", "--resume", "--limit", str(payload.limit or 1500)]
            if payload.workers:
                command += ["--workers", str(payload.workers)]
        case "label":
            command += ["label", "--resume", "--limit", str(payload.limit or 1500)]
            if payload.workers:
                command += ["--workers", str(payload.workers)]
            if payload.model:
                command += ["--model", payload.model]
        case "train-page":
            command += [
                "train",
                "--labels",
                "data/labels.final.jsonl",
                "--out",
                "data/models/page_verdict.joblib",
                "--metrics",
                "data/models/page_metrics.json",
            ]
        case "train-link":
            command += ["train-link"]
            if payload.crawl_db:
                command += ["--crawl-db", payload.crawl_db]
    return command


def db_path() -> Path:
    return DEFAULT_DB_PATH


def connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.row_factory = sqlite3.Row
    init_schema(con)
    return con


def init_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS serp_results (
            id INTEGER PRIMARY KEY,
            query TEXT NOT NULL,
            normalized_url TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            rank INTEGER NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            url TEXT NOT NULL,
            display_url TEXT NOT NULL DEFAULT '',
            snippet TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT 'serper',
            rating INTEGER CHECK (rating BETWEEN 1 AND 5),
            label TEXT CHECK (label IN ('positive', 'negative', 'skip')),
            notes TEXT NOT NULL DEFAULT '',
            pageverdict_score REAL,
            pageverdict_label TEXT,
            pageverdict_decision TEXT,
            pageverdict_model TEXT,
            crawler_source_table TEXT,
            crawler_exclusion_reason TEXT,
            crawler_status_code INTEGER,
            crawler_content_type TEXT,
            crawler_depth INTEGER,
            crawler_relevance REAL,
            crawler_token_count INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            rated_at TEXT,
            UNIQUE(query, normalized_url)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS link_results (
            id INTEGER PRIMARY KEY,
            parent_url TEXT NOT NULL,
            parent_host TEXT NOT NULL DEFAULT '',
            parent_depth INTEGER,
            parent_pageverdict_score REAL,
            parent_pageverdict_label TEXT,
            parent_pageverdict_decision TEXT,
            parent_relevance REAL,
            anchor TEXT NOT NULL DEFAULT '',
            target_url TEXT NOT NULL,
            target_host TEXT NOT NULL DEFAULT '',
            target_depth INTEGER,
            raw_score REAL,
            linkverdict_score REAL,
            linkverdict_label TEXT,
            linkverdict_model TEXT,
            should_enqueue INTEGER NOT NULL DEFAULT 0,
            selected INTEGER NOT NULL DEFAULT 0,
            rejection_reason TEXT,
            target_status TEXT,
            target_status_code INTEGER,
            target_content_type TEXT,
            target_language TEXT,
            target_relevance REAL,
            target_token_count INTEGER,
            target_pageverdict_score REAL,
            target_pageverdict_label TEXT,
            target_pageverdict_decision TEXT,
            target_exclusion_reason TEXT,
            target_fetched_at TEXT,
            source TEXT NOT NULL DEFAULT 'crawler_linkverdict',
            rating INTEGER CHECK (rating BETWEEN 1 AND 5),
            label TEXT CHECK (label IN ('positive', 'negative', 'skip')),
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            rated_at TEXT,
            UNIQUE(parent_url, target_url, anchor)
        )
        """
    )
    migrate_schema(con)
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_serp_results_query ON serp_results(query)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_serp_results_rating ON serp_results(rating)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_serp_results_source ON serp_results(source)"
    )
    con.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_serp_results_pageverdict_score
        ON serp_results(pageverdict_score)
        """
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_link_results_rating ON link_results(rating)"
    )
    con.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_link_results_target_score
        ON link_results(target_pageverdict_score)
        """
    )
    con.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_link_results_source
        ON link_results(source)
        """
    )
    con.commit()


def migrate_schema(con: sqlite3.Connection) -> None:
    columns = {
        row["name"]
        for row in con.execute("PRAGMA table_info(serp_results)").fetchall()
    }
    migrations = {
        "pageverdict_score": "REAL",
        "pageverdict_label": "TEXT",
        "pageverdict_decision": "TEXT",
        "pageverdict_model": "TEXT",
        "crawler_source_table": "TEXT",
        "crawler_exclusion_reason": "TEXT",
        "crawler_status_code": "INTEGER",
        "crawler_content_type": "TEXT",
        "crawler_depth": "INTEGER",
        "crawler_relevance": "REAL",
        "crawler_token_count": "INTEGER",
    }
    for name, definition in migrations.items():
        if name not in columns:
            con.execute(f"ALTER TABLE serp_results ADD COLUMN {name} {definition}")

    link_columns = {
        row["name"]
        for row in con.execute("PRAGMA table_info(link_results)").fetchall()
    }
    link_migrations = {
        "source": "TEXT NOT NULL DEFAULT 'crawler_linkverdict'",
        "rating": "INTEGER CHECK (rating BETWEEN 1 AND 5)",
        "label": "TEXT CHECK (label IN ('positive', 'negative', 'skip'))",
        "notes": "TEXT NOT NULL DEFAULT ''",
        "created_at": "TEXT NOT NULL DEFAULT ''",
        "updated_at": "TEXT NOT NULL DEFAULT ''",
        "rated_at": "TEXT",
        "linkverdict_score": "REAL",
        "linkverdict_label": "TEXT",
        "linkverdict_model": "TEXT",
    }
    for name, definition in link_migrations.items():
        if name not in link_columns:
            con.execute(f"ALTER TABLE link_results ADD COLUMN {name} {definition}")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_query(query: str) -> str:
    return " ".join(query.strip().split())


def normalized_url(url: str) -> str:
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    elif netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")

    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    return urlunparse((scheme, netloc, path, "", query, ""))


def label_from_rating(rating: int | None) -> LabelValue | None:
    if rating is None:
        return None
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "skip"
    return "positive"


def optional_int(value: object) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def optional_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def optional_bool(value: object) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def resolve_import_path(path_value: str) -> Path:
    raw = Path(path_value).expanduser()
    if raw.is_absolute():
        return raw
    return ROOT_DIR / raw


def serper_api_key() -> str:
    key = os.environ.get("SERPER_API_KEY", "").strip()
    if not key:
        raise HTTPException(
            status_code=503,
            detail="SERPER_API_KEY is not set",
        )
    return key


def serper_page(query: str, page_number: int) -> list[dict[str, object]]:
    body = json.dumps(
        {
            "q": query,
            "num": SERPER_COUNT,
            "page": page_number,
        }
    ).encode("utf-8")
    request = Request(
        SERPER_ENDPOINT,
        data=body,
        headers={
            "X-API-KEY": serper_api_key(),
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "LabelingLab/0.1",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=SERPER_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=exc.code, detail=detail or str(exc)) from exc
    except (TimeoutError, URLError, OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=502, detail=f"serper_search_failed: {exc}") from exc

    return payload.get("organic", [])


def fetch_serper_results(query: str) -> list[dict[str, object]]:
    seen: set[str] = set()
    results: list[dict[str, object]] = []
    for page_number in range(1, SERPER_PAGES + 1):
        for index, item in enumerate(serper_page(query, page_number), start=1):
            url = str(item.get("link", "") or "")
            if not url:
                continue
            norm = normalized_url(url)
            if norm in seen:
                continue
            seen.add(norm)
            results.append(
                {
                    "query": query,
                    "normalized_url": norm,
                    "page_number": page_number,
                    "rank": (page_number - 1) * SERPER_COUNT + index,
                    "title": item.get("title", "") or "",
                    "url": url,
                    "display_url": item.get("displayedLink", "") or "",
                    "snippet": item.get("snippet", "") or "",
                    "source": SERPER_PROVIDER,
                }
            )
    return results


def upsert_results(con: sqlite3.Connection, results: list[dict[str, object]]) -> None:
    timestamp = now()
    for result in results:
        con.execute(
            """
            INSERT INTO serp_results (
                query,
                normalized_url,
                page_number,
                rank,
                title,
                url,
                display_url,
                snippet,
                source,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(query, normalized_url) DO UPDATE SET
                page_number = excluded.page_number,
                rank = excluded.rank,
                title = excluded.title,
                url = excluded.url,
                display_url = excluded.display_url,
                snippet = excluded.snippet,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (
                result["query"],
                result["normalized_url"],
                result["page_number"],
                result["rank"],
                result["title"],
                result["url"],
                result["display_url"],
                result["snippet"],
                result["source"],
                timestamp,
                timestamp,
            ),
        )


def crawler_result_from_record(
    row: dict[str, object],
    *,
    query: str,
    fallback_rank: int,
) -> dict[str, object] | None:
    url = str(row.get("url") or "").strip()
    if not url:
        return None

    rank = optional_int(row.get("rank")) or fallback_rank
    page_number = optional_int(row.get("page")) or 1
    source = str(row.get("source") or "crawler_pageverdict").strip() or "crawler_pageverdict"
    row_query = clean_query(str(row.get("query") or ""))
    result_query = row_query if source != "crawler_pageverdict" and row_query else query

    return {
        "query": result_query,
        "normalized_url": normalized_url(url),
        "page_number": page_number,
        "rank": rank,
        "title": str(row.get("title") or ""),
        "url": url,
        "display_url": str(row.get("display_url") or urlparse(url).hostname or ""),
        "snippet": str(row.get("snippet") or ""),
        "source": source,
        "notes": str(row.get("notes") or ""),
        "pageverdict_score": optional_float(row.get("pageverdict_score")),
        "pageverdict_label": row.get("pageverdict_label") or None,
        "pageverdict_decision": row.get("pageverdict_decision") or None,
        "pageverdict_model": row.get("pageverdict_model") or None,
        "crawler_source_table": row.get("source_table") or None,
        "crawler_exclusion_reason": row.get("exclusion_reason") or None,
        "crawler_status_code": optional_int(row.get("status_code")),
        "crawler_content_type": row.get("content_type") or None,
        "crawler_depth": optional_int(row.get("crawl_depth")),
        "crawler_relevance": optional_float(row.get("relevance")),
        "crawler_token_count": optional_int(row.get("token_count")),
    }


def read_records(path: Path) -> list[dict[str, object]]:
    if path.suffix != ".jsonl":
        raise ValueError("candidate files must be JSONL")
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def read_crawler_candidates(path: Path, query: str) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for index, row in enumerate(read_records(path), start=1):
        result = crawler_result_from_record(
            row,
            query=query,
            fallback_rank=index,
        )
        if result is not None:
            results.append(result)
    return results


def link_result_from_record(
    row: dict[str, object],
) -> dict[str, object] | None:
    target_url = str(row.get("target_url") or row.get("url") or "").strip()
    if not target_url:
        return None

    parent_url = str(row.get("parent_url") or "").strip()
    anchor = str(row.get("anchor") or "")
    return {
        "parent_url": parent_url,
        "parent_host": row.get("parent_host") or urlparse(parent_url).hostname or "",
        "parent_depth": optional_int(row.get("parent_depth")),
        "parent_pageverdict_score": optional_float(row.get("parent_pageverdict_score")),
        "parent_pageverdict_label": row.get("parent_pageverdict_label") or None,
        "parent_pageverdict_decision": row.get("parent_pageverdict_decision") or None,
        "parent_relevance": optional_float(row.get("parent_relevance")),
        "anchor": anchor,
        "target_url": target_url,
        "target_host": row.get("target_host") or urlparse(target_url).hostname or "",
        "target_depth": optional_int(row.get("target_depth")),
        "raw_score": optional_float(row.get("raw_score")),
        "linkverdict_score": optional_float(row.get("linkverdict_score")),
        "linkverdict_label": row.get("linkverdict_label") or None,
        "linkverdict_model": row.get("linkverdict_model") or None,
        "should_enqueue": optional_bool(row.get("should_enqueue")),
        "selected": optional_bool(row.get("selected")),
        "rejection_reason": row.get("rejection_reason") or None,
        "target_status": row.get("target_status") or None,
        "target_status_code": optional_int(row.get("target_status_code")),
        "target_content_type": row.get("target_content_type") or None,
        "target_language": row.get("target_language") or None,
        "target_relevance": optional_float(row.get("target_relevance")),
        "target_token_count": optional_int(row.get("target_token_count")),
        "target_pageverdict_score": optional_float(row.get("target_pageverdict_score")),
        "target_pageverdict_label": row.get("target_pageverdict_label") or None,
        "target_pageverdict_decision": row.get("target_pageverdict_decision") or None,
        "target_exclusion_reason": row.get("target_exclusion_reason") or None,
        "target_fetched_at": row.get("target_fetched_at") or None,
        "source": row.get("source") or "crawler_linkverdict",
        "notes": row.get("notes") or "",
    }


def read_link_candidates(path: Path) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for row in read_records(path):
        result = link_result_from_record(row)
        if result is not None:
            results.append(result)
    return results


def review_item(row: dict[str, object]) -> dict[str, object]:
    return {
        **row,
        "title": str(row.get("title") or ""),
        "url": str(row.get("url") or ""),
        "display_url": str(row.get("display_url") or row.get("url") or ""),
        "snippet": str(row.get("snippet") or row.get("text") or ""),
        "source": "teacher_review",
        "rating": None,
    }


def upsert_crawler_results(
    con: sqlite3.Connection,
    results: list[dict[str, object]],
) -> None:
    timestamp = now()
    for result in results:
        con.execute(
            """
            INSERT INTO serp_results (
                query,
                normalized_url,
                page_number,
                rank,
                title,
                url,
                display_url,
                snippet,
                source,
                notes,
                pageverdict_score,
                pageverdict_label,
                pageverdict_decision,
                pageverdict_model,
                crawler_source_table,
                crawler_exclusion_reason,
                crawler_status_code,
                crawler_content_type,
                crawler_depth,
                crawler_relevance,
                crawler_token_count,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(query, normalized_url) DO UPDATE SET
                page_number = excluded.page_number,
                rank = excluded.rank,
                title = excluded.title,
                url = excluded.url,
                display_url = excluded.display_url,
                snippet = excluded.snippet,
                source = excluded.source,
                notes = CASE
                    WHEN serp_results.notes = '' THEN excluded.notes
                    ELSE serp_results.notes
                END,
                pageverdict_score = excluded.pageverdict_score,
                pageverdict_label = excluded.pageverdict_label,
                pageverdict_decision = excluded.pageverdict_decision,
                pageverdict_model = excluded.pageverdict_model,
                crawler_source_table = excluded.crawler_source_table,
                crawler_exclusion_reason = excluded.crawler_exclusion_reason,
                crawler_status_code = excluded.crawler_status_code,
                crawler_content_type = excluded.crawler_content_type,
                crawler_depth = excluded.crawler_depth,
                crawler_relevance = excluded.crawler_relevance,
                crawler_token_count = excluded.crawler_token_count,
                updated_at = excluded.updated_at
            """,
            (
                result["query"],
                result["normalized_url"],
                result["page_number"],
                result["rank"],
                result["title"],
                result["url"],
                result["display_url"],
                result["snippet"],
                result["source"],
                result["notes"],
                result["pageverdict_score"],
                result["pageverdict_label"],
                result["pageverdict_decision"],
                result["pageverdict_model"],
                result["crawler_source_table"],
                result["crawler_exclusion_reason"],
                result["crawler_status_code"],
                result["crawler_content_type"],
                result["crawler_depth"],
                result["crawler_relevance"],
                result["crawler_token_count"],
                timestamp,
                timestamp,
            ),
        )


def upsert_link_results(
    con: sqlite3.Connection,
    results: list[dict[str, object]],
) -> None:
    timestamp = now()
    for result in results:
        con.execute(
            """
            INSERT INTO link_results (
                parent_url,
                parent_host,
                parent_depth,
                parent_pageverdict_score,
                parent_pageverdict_label,
                parent_pageverdict_decision,
                parent_relevance,
                anchor,
                target_url,
                target_host,
                target_depth,
                raw_score,
                linkverdict_score,
                linkverdict_label,
                linkverdict_model,
                should_enqueue,
                selected,
                rejection_reason,
                target_status,
                target_status_code,
                target_content_type,
                target_language,
                target_relevance,
                target_token_count,
                target_pageverdict_score,
                target_pageverdict_label,
                target_pageverdict_decision,
                target_exclusion_reason,
                target_fetched_at,
                source,
                notes,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(parent_url, target_url, anchor) DO UPDATE SET
                parent_host = excluded.parent_host,
                parent_depth = excluded.parent_depth,
                parent_pageverdict_score = excluded.parent_pageverdict_score,
                parent_pageverdict_label = excluded.parent_pageverdict_label,
                parent_pageverdict_decision = excluded.parent_pageverdict_decision,
                parent_relevance = excluded.parent_relevance,
                target_host = excluded.target_host,
                target_depth = excluded.target_depth,
                raw_score = excluded.raw_score,
                linkverdict_score = excluded.linkverdict_score,
                linkverdict_label = excluded.linkverdict_label,
                linkverdict_model = excluded.linkverdict_model,
                should_enqueue = excluded.should_enqueue,
                selected = excluded.selected,
                rejection_reason = excluded.rejection_reason,
                target_status = excluded.target_status,
                target_status_code = excluded.target_status_code,
                target_content_type = excluded.target_content_type,
                target_language = excluded.target_language,
                target_relevance = excluded.target_relevance,
                target_token_count = excluded.target_token_count,
                target_pageverdict_score = excluded.target_pageverdict_score,
                target_pageverdict_label = excluded.target_pageverdict_label,
                target_pageverdict_decision = excluded.target_pageverdict_decision,
                target_exclusion_reason = excluded.target_exclusion_reason,
                target_fetched_at = excluded.target_fetched_at,
                source = excluded.source,
                notes = CASE
                    WHEN link_results.notes = '' THEN excluded.notes
                    ELSE link_results.notes
                END,
                updated_at = excluded.updated_at
            """,
            (
                result["parent_url"],
                result["parent_host"],
                result["parent_depth"],
                result["parent_pageverdict_score"],
                result["parent_pageverdict_label"],
                result["parent_pageverdict_decision"],
                result["parent_relevance"],
                result["anchor"],
                result["target_url"],
                result["target_host"],
                result["target_depth"],
                result["raw_score"],
                result.get("linkverdict_score"),
                result.get("linkverdict_label"),
                result.get("linkverdict_model"),
                int(bool(result["should_enqueue"])),
                int(bool(result["selected"])),
                result["rejection_reason"],
                result["target_status"],
                result["target_status_code"],
                result["target_content_type"],
                result["target_language"],
                result["target_relevance"],
                result["target_token_count"],
                result["target_pageverdict_score"],
                result["target_pageverdict_label"],
                result["target_pageverdict_decision"],
                result["target_exclusion_reason"],
                result["target_fetched_at"],
                result["source"],
                result["notes"],
                timestamp,
                timestamp,
            ),
        )


def row_to_result(row: sqlite3.Row) -> SerpResult:
    return SerpResult(
        id=row["id"],
        query=row["query"],
        page_number=row["page_number"],
        rank=row["rank"],
        title=row["title"],
        url=row["url"],
        display_url=row["display_url"],
        snippet=row["snippet"],
        source=row["source"],
        rating=row["rating"],
        label=row["label"],
        notes=row["notes"] or "",
        pageverdict_score=row["pageverdict_score"],
        pageverdict_label=row["pageverdict_label"],
        pageverdict_decision=row["pageverdict_decision"],
        pageverdict_model=row["pageverdict_model"],
        crawler_source_table=row["crawler_source_table"],
        crawler_exclusion_reason=row["crawler_exclusion_reason"],
        crawler_status_code=row["crawler_status_code"],
        crawler_content_type=row["crawler_content_type"],
        crawler_depth=row["crawler_depth"],
        crawler_relevance=row["crawler_relevance"],
        crawler_token_count=row["crawler_token_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        rated_at=row["rated_at"],
    )


def row_to_link_result(row: sqlite3.Row) -> LinkResult:
    return LinkResult(
        id=row["id"],
        parent_url=row["parent_url"],
        parent_host=row["parent_host"],
        parent_depth=row["parent_depth"],
        parent_pageverdict_score=row["parent_pageverdict_score"],
        parent_pageverdict_label=row["parent_pageverdict_label"],
        parent_pageverdict_decision=row["parent_pageverdict_decision"],
        parent_relevance=row["parent_relevance"],
        anchor=row["anchor"],
        target_url=row["target_url"],
        target_host=row["target_host"],
        target_depth=row["target_depth"],
        raw_score=row["raw_score"],
        linkverdict_score=row["linkverdict_score"],
        linkverdict_label=row["linkverdict_label"],
        linkverdict_model=row["linkverdict_model"],
        should_enqueue=bool(row["should_enqueue"]),
        selected=bool(row["selected"]),
        rejection_reason=row["rejection_reason"],
        target_status=row["target_status"],
        target_status_code=row["target_status_code"],
        target_content_type=row["target_content_type"],
        target_language=row["target_language"],
        target_relevance=row["target_relevance"],
        target_token_count=row["target_token_count"],
        target_pageverdict_score=row["target_pageverdict_score"],
        target_pageverdict_label=row["target_pageverdict_label"],
        target_pageverdict_decision=row["target_pageverdict_decision"],
        target_exclusion_reason=row["target_exclusion_reason"],
        target_fetched_at=row["target_fetched_at"],
        source=row["source"],
        rating=row["rating"],
        label=row["label"],
        notes=row["notes"] or "",
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        rated_at=row["rated_at"],
    )


def results_for_query(con: sqlite3.Connection, query: str) -> list[SerpResult]:
    rows = con.execute(
        """
        SELECT *
        FROM serp_results
        WHERE query = ?
        ORDER BY page_number, rank, id
        """,
        (query,),
    ).fetchall()
    return [row_to_result(row) for row in rows]


def link_candidates(
    con: sqlite3.Connection,
    *,
    limit: int,
    unlabeled_only: bool,
) -> list[LinkResult]:
    link_sources = ("crawler_linkverdict", "crawler_link", "curated_link")
    placeholders = ",".join("?" for _ in link_sources)
    where = f"source IN ({placeholders})"
    params: list[object] = []
    params.extend(link_sources)
    params.extend(link_sources)
    if unlabeled_only:
        where += " AND rating IS NULL"
    params.append(limit)
    rows = con.execute(
        f"""
        WITH host_stats AS (
            SELECT
                target_host,
                SUM(rating IS NOT NULL) AS rated_for_host
            FROM link_results
            WHERE source IN ({placeholders})
            GROUP BY target_host
        ),
        candidates AS (
            SELECT
                link_results.*,
                COALESCE(host_stats.rated_for_host, 0) AS rated_for_host,
                CASE
                    WHEN target_url = parent_url THEN 1
                    WHEN rejection_reason = 'seen_url' THEN 1
                    WHEN anchor = '' THEN 1
                    WHEN target_host IN (
                        'alma.uni-tuebingen.de',
                        'books.google.com',
                        'developer.wikimedia.org',
                        'd-nb.info',
                        'donate.wikimedia.org',
                        'epv-welt.uni-tuebingen.de',
                        'exchange.uni-tuebingen.de',
                        'facebook.com',
                        'fit.uni-tuebingen.de',
                        'geohack.toolforge.org',
                        'google.com',
                        'google.de',
                        'id.loc.gov',
                        'idref.fr',
                        'instagram.com',
                        'listserv.uni-tuebingen.de',
                        'maps.google.com',
                        'praxisportal.uni-tuebingen.de',
                        'stats.wikimedia.org',
                        'stay22.com',
                        'timms.uni-tuebingen.de',
                        'tiktok.com',
                        'viaf.org',
                        'web.archive.org',
                        'webmail.uni-tuebingen.de',
                        'wikimediafoundation.org',
                        'ws-export.wmcloud.org',
                        'x.com',
                        'xtools.wmcloud.org',
                        'youtube.com',
                        'youtu.be'
                    ) THEN 1
                    WHEN target_host LIKE '%.toolforge.org' THEN 1
                    WHEN target_host LIKE '%.wmcloud.org' THEN 1
                    WHEN target_host LIKE '%.archive.org' THEN 1
                    WHEN target_host LIKE '%.tiktok.com' THEN 1
                    WHEN target_host LIKE '%.wikipedia.org'
                        AND target_host NOT IN ('en.wikipedia.org', 'en.wikivoyage.org')
                    THEN 1
                    WHEN target_url LIKE '%/maps/%' THEN 1
                    WHEN target_url LIKE '%/search%' AND target_host LIKE 'google.%' THEN 1
                    WHEN target_url LIKE '%/login%' THEN 1
                    WHEN target_url LIKE '%logon%' THEN 1
                    WHEN target_url LIKE '%mailman/listinfo%' THEN 1
                    WHEN target_url LIKE '%hisinoneStartPage.faces%' THEN 1
                    WHEN target_url LIKE '%StartSearch.aspx%' THEN 1
                    WHEN lower(anchor) IN (
                        'back',
                        'barrierefreiheit',
                        'contact',
                        'contact us',
                        'cookie',
                        'cookies',
                        'datenschutz',
                        'deutsch',
                        'edit',
                        'english',
                        'facebook',
                        'home',
                        'impressum',
                        'imprint',
                        'instagram',
                        'jump to content',
                        'kontakt',
                        'login',
                        'mehr',
                        'necessary cookies',
                        'newsletter',
                        'privacy',
                        'privacy policy',
                        'privacy settings',
                        'search',
                        'share',
                        'show cookie settingshide cookie settings',
                        'sitemap',
                        'skip to content',
                        'support',
                        'terms',
                        'to top',
                        'top',
                        'twitter',
                        'x'
                    ) THEN 1
                    WHEN lower(anchor) LIKE '%cookie%' THEN 1
                    WHEN lower(anchor) LIKE '%privacy%' THEN 1
                    WHEN lower(anchor) LIKE '%datenschutz%' THEN 1
                    WHEN lower(anchor) LIKE '%impressum%' THEN 1
                    WHEN lower(anchor) LIKE '%spam prevention%' THEN 1
                    WHEN lower(anchor) LIKE 'jump %' THEN 1
                    WHEN lower(anchor) LIKE 'skip %' THEN 1
                    ELSE 0
                END AS navigation_noise,
                CASE
                    WHEN source = 'curated_link' THEN 0
                    WHEN (
                        lower(target_url) LIKE '%tuebingen%'
                        OR lower(target_url) LIKE '%tubingen%'
                        OR lower(anchor) LIKE '%tübingen%'
                        OR lower(anchor) LIKE '%tuebingen%'
                        OR lower(anchor) LIKE '%tubingen%'
                        OR target_host LIKE '%.uni-tuebingen.de'
                        OR target_host IN (
                            'bbc.com',
                            'cyber-valley.de',
                            'en.wikivoyage.org',
                            'gotouniversity.com',
                            'hoelderlinturm.de',
                            'hotelamschloss.de',
                            'kunsthalle-tuebingen.de',
                            'medizin.uni-tuebingen.de',
                            'my-stuwe.de',
                            'naturpark-schoenbuch.de',
                            'tuebingenresearchcampus.com',
                            'www.medizin.uni-tuebingen.de'
                        )
                    )
                    AND (
                        lower(target_url) LIKE '%/en%'
                        OR lower(target_url) LIKE '%english%'
                        OR lower(anchor) GLOB '*[A-Za-z]*'
                        OR target_host IN ('bbc.com', 'en.wikipedia.org', 'en.wikivoyage.org')
                    )
                    THEN 0
                    WHEN selected = 1 AND should_enqueue = 1 THEN 1
                    WHEN should_enqueue = 1 THEN 2
                    ELSE 3
                END AS candidate_priority,
                CASE
                    WHEN raw_score IS NULL THEN 4
                    WHEN raw_score >= 8 THEN 0
                    WHEN raw_score >= 4 THEN 1
                    WHEN raw_score >= 2 THEN 2
                    ELSE 3
                END AS score_band,
                CASE
                    WHEN linkverdict_score IS NULL THEN 3
                    WHEN linkverdict_score >= 0.75 THEN 0
                    WHEN linkverdict_score >= 0.40 THEN 1
                    ELSE 2
                END AS linkverdict_band,
                CASE
                    WHEN target_pageverdict_score IS NULL THEN 1.0
                    ELSE ABS(target_pageverdict_score - 0.5)
                END AS target_uncertainty,
                CASE
                    WHEN linkverdict_score IS NULL THEN 1.0
                    ELSE ABS(linkverdict_score - 0.5)
                END AS link_uncertainty,
                CASE
                    WHEN source = 'curated_link' THEN 0
                    ELSE 1
                END AS source_priority
            FROM link_results
            LEFT JOIN host_stats USING (target_host)
            WHERE {where}
        ),
        ranked AS (
            SELECT
                candidates.*,
                ROW_NUMBER() OVER (
                    PARTITION BY target_host
                    ORDER BY
                        CASE WHEN rating IS NULL THEN 0 ELSE 1 END,
                        target_uncertainty,
                        selected DESC,
                        should_enqueue DESC,
                        raw_score DESC,
                        id
                ) AS host_round,
                ROW_NUMBER() OVER (
                    PARTITION BY target_url
                    ORDER BY
                        CASE WHEN rating IS NULL THEN 0 ELSE 1 END,
                        target_uncertainty,
                        selected DESC,
                        should_enqueue DESC,
                        raw_score DESC,
                        id
                ) AS url_round,
                ROW_NUMBER() OVER (
                    PARTITION BY score_band
                    ORDER BY
                        CASE WHEN rating IS NULL THEN 0 ELSE 1 END,
                        target_uncertainty,
                        selected DESC,
                        should_enqueue DESC,
                        raw_score DESC,
                        id
                ) AS score_band_round
            FROM candidates
        )
        SELECT *
        FROM ranked
        ORDER BY
            source_priority,
            navigation_noise,
            candidate_priority,
            linkverdict_band,
            url_round,
            host_round,
            score_band_round,
            rated_for_host,
            link_uncertainty,
            score_band,
            target_uncertainty,
            selected DESC,
            should_enqueue DESC,
            raw_score DESC,
            id
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [row_to_link_result(row) for row in rows]


def crawler_candidates(
    con: sqlite3.Connection,
    *,
    limit: int,
    unlabeled_only: bool,
    sources: tuple[str, ...] = DEFAULT_CANDIDATE_SOURCES,
) -> list[SerpResult]:
    source_values = tuple(source for source in sources if source)
    if not source_values:
        return []

    placeholders = ",".join("?" for _ in source_values)
    where = f"source IN ({placeholders}) AND pageverdict_score IS NOT NULL"
    params: list[object] = list(source_values)
    if unlabeled_only:
        where += " AND rating IS NULL"
    params.append(limit)
    rows = con.execute(
        f"""
        SELECT *
        FROM serp_results
        WHERE {where}
        ORDER BY
            CASE WHEN pageverdict_decision LIKE 'false_%' THEN 0 ELSE 1 END,
            ABS(pageverdict_score - 0.5),
            pageverdict_score,
            rank,
            id
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [row_to_result(row) for row in rows]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/search", response_model=SearchResponse)
def search(payload: SearchRequest) -> SearchResponse:
    query = clean_query(payload.query)
    if not query:
        raise HTTPException(status_code=400, detail="query_required")

    results = fetch_serper_results(query)
    with connect() as con:
        with con:
            upsert_results(con, results)
        stored = results_for_query(con, query)

    return SearchResponse(
        query=query,
        source=SERPER_PROVIDER,
        pages=SERPER_PAGES,
        count_per_page=SERPER_COUNT,
        results=stored,
    )


@app.get("/api/results", response_model=list[SerpResult])
def results(query: str) -> list[SerpResult]:
    query = clean_query(query)
    if not query:
        return []
    with connect() as con:
        return results_for_query(con, query)


@app.post("/api/import/crawler-pageverdict", response_model=CrawlerImportResponse)
def import_crawler_pageverdict(payload: CrawlerImportRequest) -> CrawlerImportResponse:
    query = clean_query(payload.query)
    if not query:
        raise HTTPException(status_code=400, detail="query_required")

    path = resolve_import_path(payload.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"candidate_file_not_found: {path}")
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"not_a_file: {path}")

    try:
        imported = read_crawler_candidates(path, query)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid_candidate_file: {exc}") from exc

    with connect() as con:
        with con:
            upsert_crawler_results(con, imported)
        sources = tuple(sorted({str(result["source"]) for result in imported if result.get("source")}))
        results = crawler_candidates(
            con,
            limit=payload.limit,
            unlabeled_only=payload.unlabeled_only,
            sources=sources or DEFAULT_CANDIDATE_SOURCES,
        )

    return CrawlerImportResponse(
        path=str(path),
        rows_read=len(imported),
        stored=len(imported),
        results=results,
    )


@app.get("/api/crawler-candidates", response_model=list[SerpResult])
def get_crawler_candidates(
    limit: int = 200,
    unlabeled_only: bool = True,
) -> list[SerpResult]:
    limit = max(1, min(limit, 2000))
    with connect() as con:
        return crawler_candidates(con, limit=limit, unlabeled_only=unlabeled_only)


@app.post("/api/import/link-candidates", response_model=LinkImportResponse)
def import_link_candidates(payload: LinkImportRequest) -> LinkImportResponse:
    path = resolve_import_path(payload.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"candidate_file_not_found: {path}")
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"not_a_file: {path}")

    try:
        imported = read_link_candidates(path)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid_candidate_file: {exc}") from exc

    with connect() as con:
        with con:
            upsert_link_results(con, imported)
        results = link_candidates(
            con,
            limit=payload.limit,
            unlabeled_only=payload.unlabeled_only,
        )

    return LinkImportResponse(
        path=str(path),
        rows_read=len(imported),
        stored=len(imported),
        results=results,
    )


@app.get("/api/link-candidates", response_model=list[LinkResult])
def get_link_candidates(
    limit: int = 200,
    unlabeled_only: bool = True,
) -> list[LinkResult]:
    limit = max(1, min(limit, 2000))
    with connect() as con:
        return link_candidates(con, limit=limit, unlabeled_only=unlabeled_only)


@app.post("/api/import/review-batch")
def import_review_batch(payload: ReviewImportRequest) -> dict[str, object]:
    path = resolve_import_path(payload.path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"review_batch_not_found: {path}")
    try:
        items = [review_item(row) for row in read_records(path)]
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid_review_batch: {exc}") from exc
    return {"path": str(path), "rows_read": len(items), "results": items}


@app.post("/api/rating", response_model=ActionResponse)
def rate(payload: RatingRequest) -> ActionResponse:
    label = label_from_rating(payload.rating)
    timestamp = now()
    with connect() as con:
        with con:
            row = con.execute(
                "SELECT id FROM serp_results WHERE id = ?",
                (payload.result_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="result_not_found")
            con.execute(
                """
                UPDATE serp_results
                SET
                    rating = ?,
                    label = ?,
                    notes = ?,
                    rated_at = CASE WHEN ? IS NULL THEN NULL ELSE ? END,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    payload.rating,
                    label,
                    payload.notes,
                    payload.rating,
                    timestamp,
                    timestamp,
                    payload.result_id,
                ),
            )
    return ActionResponse()


@app.post("/api/link-rating", response_model=ActionResponse)
def rate_link(payload: LinkRatingRequest) -> ActionResponse:
    label = label_from_rating(payload.rating)
    timestamp = now()
    with connect() as con:
        with con:
            row = con.execute(
                "SELECT id FROM link_results WHERE id = ?",
                (payload.link_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="link_result_not_found")
            con.execute(
                """
                UPDATE link_results
                SET
                    rating = ?,
                    label = ?,
                    notes = ?,
                    rated_at = CASE WHEN ? IS NULL THEN NULL ELSE ? END,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    payload.rating,
                    label,
                    payload.notes,
                    payload.rating,
                    timestamp,
                    timestamp,
                    payload.link_id,
                ),
            )
    return ActionResponse()


@app.post("/api/review-rating", response_model=ActionResponse)
def rate_review_item(payload: ReviewRatingRequest) -> ActionResponse:
    item = dict(payload.item)
    item.update(
        {
            "teacher_label": item.get("label"),
            "label": label_from_rating(payload.rating),
            "rating": payload.rating,
            "review_notes": payload.notes,
            "reviewed_at": utc_now(),
            "review_source": "labeling_lab_ui",
        }
    )
    append_jsonl(ROOT_DIR / "data" / "labels.reviewed.jsonl", item)
    return ActionResponse()


@app.get("/api/stats", response_model=LabelStats)
def stats() -> LabelStats:
    with connect() as con:
        total_row = con.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(rating IS NOT NULL) AS rated
            FROM serp_results
            """
        ).fetchone()
        label_rows = con.execute(
            "SELECT label, COUNT(*) AS count FROM serp_results WHERE label IS NOT NULL GROUP BY label"
        ).fetchall()
        rating_rows = con.execute(
            "SELECT rating, COUNT(*) AS count FROM serp_results WHERE rating IS NOT NULL GROUP BY rating"
        ).fetchall()
    return LabelStats(
        results=total_row["total"] or 0,
        rated=total_row["rated"] or 0,
        labels={row["label"]: row["count"] for row in label_rows},
        ratings={str(row["rating"]): row["count"] for row in rating_rows},
    )


@app.get("/api/link-stats", response_model=LabelStats)
def link_stats() -> LabelStats:
    with connect() as con:
        total_row = con.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(rating IS NOT NULL) AS rated
            FROM link_results
            """
        ).fetchone()
        label_rows = con.execute(
            "SELECT label, COUNT(*) AS count FROM link_results WHERE label IS NOT NULL GROUP BY label"
        ).fetchall()
        rating_rows = con.execute(
            "SELECT rating, COUNT(*) AS count FROM link_results WHERE rating IS NOT NULL GROUP BY rating"
        ).fetchall()
    return LabelStats(
        results=total_row["total"] or 0,
        rated=total_row["rated"] or 0,
        labels={row["label"]: row["count"] for row in label_rows},
        ratings={str(row["rating"]): row["count"] for row in rating_rows},
    )


@app.get("/api/pipeline-status")
def get_pipeline_status() -> dict[str, object]:
    return pipeline_status()


@app.post("/api/pipeline-run")
def run_pipeline(payload: PipelineRequest) -> dict[str, object]:
    with JOB.lock:
        if JOB.process is not None and JOB.process.poll() is None:
            raise HTTPException(status_code=409, detail="pipeline_job_already_running")
        command = _pipeline_command(payload)
        JOB.command = " ".join(command)
        JOB.output = [f"$ {JOB.command}"]
        JOB.process = subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        threading.Thread(target=_read_job_output, args=(JOB.process,), daemon=True).start()
    return pipeline_status()


def jsonl_response(
    filename: str,
    rows: list[sqlite3.Row],
) -> StreamingResponse:
    buffer = io.StringIO()
    for row in rows:
        buffer.write(json.dumps({key: row[key] for key in row.keys()}, ensure_ascii=False))
        buffer.write("\n")
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/export/serp-labels.jsonl")
def export_serp_labels() -> StreamingResponse:
    with connect() as con:
        rows = con.execute(
            """
            SELECT
                query,
                page_number,
                rank,
                title,
                url,
                display_url,
                snippet,
                source,
                rating,
                label,
                notes,
                pageverdict_score,
                pageverdict_label,
                pageverdict_decision,
                pageverdict_model,
                crawler_source_table,
                crawler_exclusion_reason,
                crawler_status_code,
                crawler_content_type,
                crawler_depth,
                crawler_relevance,
                crawler_token_count,
                created_at,
                updated_at,
                rated_at
            FROM serp_results
            ORDER BY query, page_number, rank, id
            """
        ).fetchall()
    return jsonl_response("serp-labels.jsonl", rows)


@app.get("/api/export/link-labels.jsonl")
def export_link_labels() -> StreamingResponse:
    with connect() as con:
        rows = con.execute(
            """
            SELECT
                parent_url,
                parent_host,
                parent_depth,
                parent_pageverdict_score,
                parent_pageverdict_label,
                parent_pageverdict_decision,
                parent_relevance,
                anchor,
                target_url,
                target_host,
                target_depth,
                raw_score,
                linkverdict_score,
                linkverdict_label,
                linkverdict_model,
                should_enqueue,
                selected,
                rejection_reason,
                target_status,
                target_status_code,
                target_content_type,
                target_language,
                target_relevance,
                target_token_count,
                target_pageverdict_score,
                target_pageverdict_label,
                target_pageverdict_decision,
                target_exclusion_reason,
                target_fetched_at,
                source,
                rating,
                label,
                notes,
                created_at,
                updated_at,
                rated_at
            FROM link_results
            ORDER BY target_url, parent_url, id
            """
        ).fetchall()
    return jsonl_response("link-labels.jsonl", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the labeling UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
