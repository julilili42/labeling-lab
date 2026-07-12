"""Tunnel-Sweep: findet englische Seiten, die hinter abgelehnten
nicht-englischen Seiten versteckt sind (Ein-Hop-Tunneling als Batch-Skript,
ohne Crawler-Änderung).

Liest die Crawl-DB read-only, fetcht non_english-Rejects erneut (max. 3 pro
Host), extrahiert unbekannte Links mit Englisch-Mustern (/en/, /english, en.*)
und schreibt Kandidaten nach data/tunnel_candidates.json.

Wiederholbar: alles bereits Bekannte (Accepts, Rejects, Link-Kandidaten) wird
übersprungen — jeder Lauf liefert nur Neues seit dem letzten Crawl-Stand.

    python3 tools/tunnel_sweep.py
    # danach: Kandidaten nach Tübingen-Bezug filtern, beste als Seeds
    # in search-engine/crawl/seeds.toml, Crawler neu starten.
"""
from __future__ import annotations

import json
import re
import sqlite3
import urllib.request
from collections import defaultdict
from pathlib import Path
from urllib.parse import urljoin, urlsplit

CRAWL_DB = Path("/Users/julian/Documents/Projekte/search-engine/data/pages.sqlite")
OUT = Path(__file__).resolve().parents[1] / "data" / "tunnel_candidates.json"

SKIP_HOST = re.compile(r"wik(i|t)|wikimedia|fandom|facebook|youtube|instagram", re.I)
SKIP_PATH = re.compile(r"action=|Template:|Special:|index\.php|\.pdf$|\.jpg$", re.I)
EN = re.compile(r"/en([/._-]|$)|/english", re.I)
HREF = re.compile(r"href=[\"']([^\"'#]+)[\"']", re.I)
MAX_PER_HOST = 3


def main() -> None:
    db = sqlite3.connect(f"file:{CRAWL_DB}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    known = {r[0] for r in db.execute("SELECT url FROM pages")}
    known |= {r[0] for r in db.execute("SELECT url FROM rejected_pages")}
    known |= {r[0] for r in db.execute("SELECT target_url FROM link_candidates")}
    rows = db.execute(
        "SELECT url, host FROM rejected_pages WHERE exclusion_reason='non_english'"
    ).fetchall()

    per_host: dict[str, int] = defaultdict(int)
    sample = []
    for r in rows:
        if SKIP_HOST.search(r["host"] or "") or per_host[r["host"]] >= MAX_PER_HOST:
            continue
        per_host[r["host"]] += 1
        sample.append(r["url"])
    print(f"zu fetchen: {len(sample)}", flush=True)

    found: set[str] = set()
    for i, url in enumerate(sample):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (tunnel-sweep)"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                html = resp.read(500_000).decode("utf-8", errors="replace")
        except Exception:
            continue
        for href in HREF.findall(html):
            absu = urljoin(url, href.replace("&amp;", "&"))
            if not absu.startswith("http") or absu in known:
                continue
            host = urlsplit(absu).hostname or ""
            if SKIP_HOST.search(host) or SKIP_PATH.search(absu):
                continue
            if EN.search(urlsplit(absu).path) or host.startswith("en."):
                found.add(absu)
        if i and i % 100 == 0:
            print(f"{i} gefetcht, {len(found)} Kandidaten", flush=True)

    OUT.write_text(json.dumps(sorted(found), indent=1))
    hosts = {urlsplit(u).hostname for u in found}
    print(f"FERTIG: {len(found)} Kandidaten auf {len(hosts)} Hosts -> {OUT}")


if __name__ == "__main__":
    main()
