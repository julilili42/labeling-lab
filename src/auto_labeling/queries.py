from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_RESULTS_PER_QUERY = 20
MAX_RESULTS_PER_QUERY = 100


@dataclass(frozen=True)
class QuerySpec:
    query: str
    results: int = DEFAULT_RESULTS_PER_QUERY


def parse_query_line(line: str) -> QuerySpec | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "|" not in line:
        return QuerySpec(query=line)
    query, raw_count = [part.strip() for part in line.split("|", 1)]
    count = min(MAX_RESULTS_PER_QUERY, max(1, int(raw_count)))
    return QuerySpec(query=query, results=count)


def read_queries(path: Path, *, limit: int | None = None) -> list[QuerySpec]:
    specs: list[QuerySpec] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        spec = parse_query_line(line)
        if spec is None:
            continue
        specs.append(spec)
        if limit is not None and len(specs) >= limit:
            break
    return specs
