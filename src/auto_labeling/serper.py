from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .urls import normalize_url

SERPER_ENDPOINT = "https://google.serper.dev/search"
SERPER_PAGE_SIZE = 10


def serper_page(
    query: str,
    *,
    page: int,
    num: int,
    api_key: str,
    timeout: int = 20,
) -> list[dict[str, object]]:
    payload = json.dumps({"q": query, "num": num, "page": page}).encode("utf-8")
    request = Request(
        SERPER_ENDPOINT,
        data=payload,
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"serper_search_failed: {exc}") from exc
    organic = data.get("organic", [])
    return organic if isinstance(organic, list) else []


def dedupe_serp_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_url: dict[str, dict[str, object]] = {}
    for row in rows:
        normalized = str(row.get("normalized_url") or normalize_url(str(row.get("url") or "")))
        existing = by_url.get(normalized)
        query = str(row.get("query") or "")
        if existing is None:
            copy = dict(row)
            copy["normalized_url"] = normalized
            copy["discovery_queries"] = [query] if query else []
            copy["best_rank"] = int(row.get("rank") or 0)
            by_url[normalized] = copy
            continue
        queries = list(existing.get("discovery_queries") or [])
        if query and query not in queries:
            queries.append(query)
        existing["discovery_queries"] = queries
        rank = int(row.get("rank") or 999999)
        existing["best_rank"] = min(int(existing.get("best_rank") or rank), rank)
    return sorted(by_url.values(), key=lambda row: (int(row.get("best_rank") or 999999), str(row.get("normalized_url") or "")))
