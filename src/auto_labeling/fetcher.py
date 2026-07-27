from __future__ import annotations

import http.client
import re
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .jsonl import stable_hash, utc_now
from .urls import host_from_url, normalize_url

MAX_BODY_BYTES = 2_000_000
USER_AGENT = "AutoLabelingBot/0.1 (+offline labeling dataset builder)"


class PageTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.description = ""
        self.h1 = ""
        self.h2: list[str] = []
        self.html_lang = ""
        self.text_parts: list[str] = []
        self._skip_depth = 0
        self._tag_stack: list[str] = []
        self._capture_title = False
        self._capture_h1 = False
        self._capture_h2 = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "html":
            self.html_lang = dict(attrs).get("lang") or ""
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        self._tag_stack.append(tag)
        if tag == "title":
            self._capture_title = True
        elif tag == "h1":
            self._capture_h1 = True
        elif tag == "h2":
            self._capture_h2 = True
        elif tag == "meta":
            attr = {key.lower(): value or "" for key, value in attrs}
            if attr.get("name", "").lower() == "description" and not self.description:
                self.description = attr.get("content", "")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._capture_title = False
        elif tag == "h1":
            self._capture_h1 = False
        elif tag == "h2":
            self._capture_h2 = False
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        text = collapse_space(data)
        if not text or self._skip_depth:
            return
        if self._capture_title and not self.title:
            self.title = text
        elif self._capture_h1 and not self.h1:
            self.h1 = text
        elif self._capture_h2 and len(self.h2) < 20:
            self.h2.append(text)
        self.text_parts.append(text)


def collapse_space(value: str) -> str:
    return " ".join(value.split())


def normalized_text_hash(text: str) -> str:
    return stable_hash(collapse_space(text).lower())


def parse_html(html: str) -> dict[str, object]:
    parser = PageTextParser()
    parser.feed(html)
    text = collapse_space(" ".join(parser.text_parts))
    return {
        "title": collapse_space(parser.title),
        "description": collapse_space(parser.description),
        "h1": collapse_space(parser.h1),
        "h2": [collapse_space(item) for item in parser.h2 if collapse_space(item)],
        "html_lang": parser.html_lang.strip().lower(),
        "text": text,
        "text_hash": normalized_text_hash(text),
    }


def _charset(content_type: str) -> str:
    match = re.search(r"charset=([^;\s]+)", content_type, flags=re.I)
    return match.group(1) if match else "utf-8"


def fetch_snapshot(row: dict[str, object], *, timeout: int = 15) -> dict[str, object]:
    url = str(row.get("url") or "")
    normalized = str(row.get("normalized_url") or normalize_url(url))
    request = Request(url, headers={"Accept-Language": "en", "User-Agent": USER_AGENT})
    base = {
        "url": url,
        "normalized_url": normalized,
        "host": host_from_url(normalized),
        "discovery_queries": row.get("discovery_queries") or [row.get("query")],
        "serp_title": row.get("title") or "",
        "serp_snippet": row.get("snippet") or "",
        "fetched_at": utc_now(),
    }
    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            status = getattr(response, "status", 200)
            body = response.read(MAX_BODY_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, ValueError, OSError, http.client.HTTPException) as exc:
        return {**base, "status_code": 0, "content_type": "", "fetch_error": str(exc)}

    if len(body) > MAX_BODY_BYTES:
        return {**base, "status_code": status, "content_type": content_type, "fetch_error": "body_too_large"}
    if "html" not in content_type.lower():
        return {**base, "status_code": status, "content_type": content_type, "fetch_error": "non_html"}

    html = body.decode(_charset(content_type), errors="replace")
    parsed = parse_html(html)
    text_hash = str(parsed.get("text_hash") or "")
    return {
        **base,
        **parsed,
        "id": stable_hash({"normalized_url": normalized, "text_hash": text_hash}),
        "status_code": status,
        "content_type": content_type,
        "fetch_error": "",
    }
