from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid"}


def normalize_host(host: str | None) -> str:
    return (host or "").lower().removeprefix("www.")


def normalize_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    scheme = (parsed.scheme or "https").lower()
    host = normalize_host(parsed.hostname)
    if not host:
        return url.strip()
    port = f":{parsed.port}" if parsed.port else ""
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in TRACKING_PARAMS
    ]
    query = urlencode(sorted(query_items), doseq=True)
    return urlunsplit((scheme, host + port, path, query, ""))


def host_from_url(url: str) -> str:
    return normalize_host(urlsplit(url).hostname)
