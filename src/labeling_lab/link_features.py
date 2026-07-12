from dataclasses import dataclass
import re
from urllib.parse import unquote, urlparse


@dataclass(frozen=True)
class LinkVerdictInput:
    anchor: str
    target_url: str
    parent_url: str = ""
    parent_host: str = ""
    parent_depth: int | None = None
    parent_pageverdict_score: float | None = None
    parent_pageverdict_decision: str = ""
    parent_relevance: float | None = None
    target_host: str = ""
    target_depth: int | None = None
    raw_score: float | None = None


def normalize_host(host: str | None) -> str:
    return (host or "").lower().removeprefix("www.")


def host_from_url(url: str) -> str:
    try:
        return normalize_host(urlparse(url).hostname)
    except ValueError:
        return ""


def _path_text(url: str) -> str:
    try:
        path = unquote(urlparse(url).path).lower()
    except ValueError:
        path = ""
    return " ".join(part for part in re.split(r"[/_.:-]+", path) if part)


def _space(value: str | None) -> str:
    return " ".join((value or "").split())


def _float(value: float | None, step: float) -> str:
    return "missing" if value is None else f"{round(value / step) * step:.2f}"


def _hard_skipable(url: str) -> bool:
    host = host_from_url(url)
    blocked_hosts = {
        "web.archive.org", "archive.today", "archive.is", "archive.ph", "atlasobscura.com",
        "bsky.app", "bsky.social", "facebook.com", "google.com", "google.de", "instagram.com",
        "linkedin.com", "maps.google.com", "maps.google.de", "pinterest.com", "pinterest.de",
        "plus.google.com", "tiktok.com", "x.com", "twitter.com", "www.google.com",
        "www.google.de", "www.atlasobscura.com", "youtube.com", "youtu.be",
    }
    if host in blocked_hosts or host.endswith((
        ".facebook.com", ".instagram.com", ".linkedin.com", ".pinterest.com", ".pinterest.de",
        ".tiktok.com", ".twitter.com", ".youtube.com", ".atlasobscura.com",
    )):
        return True
    if url.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".svg", ".css", ".js", ".pdf", ".zip", ".mp4", ".mp3", ".ico", ".woff", ".woff2", ".webp")):
        return True
    try:
        path = unquote(urlparse(url).path).lower()
        first = next((part for part in path.split("/") if part), "")
    except ValueError:
        path, first = "", ""
    if re.split(r"[-_]", first)[0] in {"cn", "de", "deutsch", "es", "fr", "it", "ja", "nl", "pl", "pt", "ru", "zh"}:
        return True
    return bool(set(re.findall(r"[a-z]+", path)) & {
        "category", "appendix", "talk", "special", "contact", "cookie", "cookies", "impressum",
        "imprint", "kontakt", "login", "logout", "newsletter", "privacy", "search", "sitemap",
        "tag", "tags", "terms",
    })


def make_text(example: LinkVerdictInput) -> str:
    target_host = normalize_host(example.target_host) or host_from_url(example.target_url)
    parent_host = normalize_host(example.parent_host) or host_from_url(example.parent_url)
    parts = [
        f"anchor: {_space(example.anchor)}",
        f"target_url: {_space(example.target_url)}",
        f"target_host: {target_host}",
        f"target_path: {_path_text(example.target_url)}",
        f"parent_url: {_space(example.parent_url)}",
        f"parent_host: {parent_host}",
        f"parent_depth: {'missing' if example.parent_depth is None else example.parent_depth}",
        f"target_depth: {'missing' if example.target_depth is None else example.target_depth}",
        f"parent_pageverdict_score_bucket: {_float(example.parent_pageverdict_score, 0.1)}",
        f"parent_pageverdict_decision: {_space(example.parent_pageverdict_decision)}",
        f"parent_relevance_bucket: {_float(example.parent_relevance, 0.5)}",
        f"flags: hard_skipable_url:{'yes' if _hard_skipable(example.target_url) else 'no'}",
    ]
    return "\n".join(parts)
