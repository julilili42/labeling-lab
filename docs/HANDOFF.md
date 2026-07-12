# Handoff: Auto Labeling / PageVerdict

Stand: 2026-07-10, Mittag. Dieses Dokument ist der Einstiegspunkt für die
Übernahme der Arbeit. **Kurzfassung: Datensammlung und erstes Modell sind
fertig — siehe "Ergebnisse & Fazit" unten.** Ergänzend: [SPEC.md](../SPEC.md) (Pipeline-Spec),
[DECISIONS.md](DECISIONS.md) (ADRs), [KNOWLEDGE.md](KNOWLEDGE.md).

## Kontext

Dieses Repo baut den gelabelten Datensatz und den PageVerdict-Klassifikator
("ist diese Seite englischer, allgemein nützlicher Tübingen-Content?") für das
Gruppenprojekt **INFO4271 Modern Search Engines (SoSe 2026)**. Das eigentliche
Suchmaschinen-Repo ist `/Users/julian/Documents/Projekte/search-engine`
(dort liegt auch `docs/Project Instructions.pdf` — die verbindlichen
Kurs-Anforderungen).

Pipeline: `queries.txt -> Serper SERP -> fetch snapshots -> teacher labels
(ollama qwen2.5:7b) -> human review -> train/evaluate Studentenmodell`.
Alles file-basiert (JSONL in `data/`), alle teuren Schritte mit `--resume`.

## Kurs-Requirements (aus Project Instructions.pdf), soweit hier relevant

- **Nur Python.** Allgemeine ML/NLP-Libs erlaubt (scikit-learn, numpy, spacy,
  nltk …). Verboten: dedizierte Crawling-/Search-Toolkits (scrapy, whoosh,
  lucene, terrier, galago …). requests/beautifulsoup sind ok.
- **Pretrained-Modelle nur, wenn sie NICHT für Retrieval gebaut/trainiert
  wurden** (Deliverables-Abschnitt + FAQ 5). Konsequenz:
  - **NICHT erlaubt:** BGE-M3, E5, nomic-embed und praktisch alle
    MTEB-Leaderboard-Retrieval-Embedder (MS-MARCO-trainiert).
  - **Erlaubt:** vanilla BERT/XLM-RoBERTa/DistilBERT (Mean-Pooling),
    Paraphrase-Sentence-Transformers (z. B.
    `paraphrase-multilingual-MiniLM-L12-v2`), general-purpose LLMs
    (der Teacher qwen2.5:7b ist ok).
- **Nur englische Inhalte sind relevant** (FAQ 2). Nicht-englische Seiten
  gelten in der Evaluation als irrelevant — deshalb ist die
  Sprachklassifikation im Label kritisch.
- **First-Stage-Retrieval muss selbst implementierte Klassik sein** (BM25
  o. ä.); Embeddings/Neural nur als Second-Stage-Re-Ranker (FAQ 3/4). Betrifft
  das search-engine-Repo; PageVerdict filtert nur den Index und rankt nicht.
- Re-Ranker selbst trainieren ist erlaubt, fine-getunte Pretrained-Re-Ranker
  nicht (FAQ 6).
- Abgabe: 4-Seiten-Report + Code-Repo (eingefrorener Branch) + UI. Bewertung
  u. a. nDCG auf 5 Queries (2 bekannt: "tübingen attractions", "food and
  drinks"; 3 weitere live am **06.08.2026**, Ziel ≤ ~1 min/Query).

## Datenstand (2026-07-10, final)

| Datei | Stand |
|---|---|
| `data/queries.txt` | 190 Queries, alle gesucht (3 Kategorien: Core positives, Research drift traps, Non-English/nearby negatives) |
| `data/serp_results.jsonl` | 3778 Zeilen, 2858 eindeutige URLs |
| `data/page_snapshots.jsonl` | 2682 Snapshots, davon 2254 fetchbare HTML (Rest: 403-Bot-Blocks, PDFs, Timeouts) |
| `data/teacher_labels.raw.jsonl` | 2253 Labels vom 7b-Teacher |
| `data/teacher_labels.14b.jsonl` | 2253 Zweit-Labels vom 14b-Teacher (nur für den Teacher-Vergleich) |
| `data/labels.reviewed.jsonl` | 476 menschlich reviewte Labels (161 Stichprobe + 315 Teacher-Konflikte) |
| `data/labels.final.jsonl` | **Trainingsdatensatz**: Teacher-Labels, durch Reviews überschrieben (259 Korrekturen), angereichert mit Seitentext, ohne Holdout |
| `data/eval_holdout.jsonl` | **Holdout**: 150 zufällige reviewte Beispiele, nie im Training |
| `data/models/page_verdict.joblib` | Finales Modell (TF-IDF + LogReg inkl. Seitentext) |
| `data/review_batches/` | batch-0001 … batch-0005 + batch-0006-conflicts |

## Ergebnisse & Fazit (2026-07-10)

**Teacher-Vergleich (7b vs. 14b, kompletter Korpus doppelt gelabelt):**
- Übereinstimmung 86 % (1938/2253); 315 Konflikte, alle von Hand reviewt.
- Am Holdout: 7b 95,0 % korrekt, 14b 87,6 %. Bei Konflikten lag 7b in 16 von
  21 Fällen richtig. **qwen2.5:7b bleibt der Teacher; 14b ist systematisch zu
  großzügig mit positive** (183× negative→positive geflippt, meist zu Unrecht).
- Der Wert des Doppel-Labelings lag in der Konfliktliste: 259 echte
  Label-Korrekturen durch gezielte Review statt Zufallsstichprobe.

**Modellvergleich (gleicher Holdout, 150 menschlich verifizierte Beispiele):**

| Variante | Holdout-Accuracy |
|---|---|
| TF-IDF (Titel/URL/Snippet) | 77,3 % |
| **TF-IDF + Seitentext (3000 Zeichen) — GEWINNER** | **78,7 %** |
| MiniLM-Embeddings (paraphrase-multilingual) + LogReg | 68,0 % |

- Die Embedding-Variante verliert klar: MiniLM kappt bei ~128 Tokens, und
  die Aufgabe hängt an Oberflächen-Signalen (Sprache, "dissertation",
  `/en/`-URLs), die TF-IDF-Wort-/Zeichen-n-Gramme direkt einfangen.
  Stärkere Encoder (BGE-M3/E5) sind durch die Kurs-Regel gesperrt.
  Code liegt in `src/auto_labeling/train_embed.py`, Schiene ist abgehakt.
- Wichtig beim Lesen der 78,7 %: Der Holdout besteht überproportional aus
  Teacher-Konfliktfällen (den schwersten Beispielen). Auf typischen Seiten
  liegt das Modell eher bei der internen Validation (~85 %).
- Bekannte Schwäche: negative-Precision 68 % im Holdout — schwierige
  positive Seiten werden zu oft verworfen.

**Vergleich mit dem alten handgelabelten Modell (search-engine/verdict-ml):**

Das alte PageVerdict wurde auf 1594 Hand-Labels trainiert (Ratings aus
`labeling.sqlite`; die DB lag im gelöschten Ordner `tuebingen-search-engine-main`,
aus dem Papierkorb wiederhergestellt, Kopie hier: `data/hand_labeling.sqlite`,
Export: `data/labels.hand.jsonl`). Alle Modelle am selben 150er-Holdout
(0 URL-Überlappung mit dem Hand-Training, sklearn-Versionen inkompatibel —
Hand-Modell wurde in der uv-Umgebung des search-engine-Repos ausgewertet):

| Modell | Trainingsdaten | Holdout |
|---|---|---|
| **Auto-only (FINAL)** | 2102 Auto-Labels + Reviews, mit Text | **78,7 %** |
| Altes Hand-Modell | 1594 Hand-Labels, ohne Text | 69,3 % |
| Merged | 3398 (Auto + Hand, Hand gewinnt bei 132 URL-Kollisionen) | 77,3 % |

Fazit: **Auto-only bleibt das finale Modell.** Die Hand-Labels verbessern
nichts — sie haben kein Text-Feld und folgen einer leicht anderen
Label-Politik, was widersprüchliche Trainingssignale erzeugt. Der Merge ist
als begründet verworfenes Experiment dokumentiert (`labels.merged.jsonl`,
`page_verdict_merged.joblib` + Metriken); fürs Crawler-Deployment gilt
`data/models/page_verdict.joblib`. Fußnote für den Report: Holdout stammt
aus der Verteilung/Label-Politik dieses Repos — auf der alten
Query-Verteilung könnte der Merge anders aussehen.

**Reproduzieren:**

```bash
PYTHONPATH=src python -m auto_labeling.cli train --labels data/labels.final.jsonl
PYTHONPATH=src python -m auto_labeling.cli evaluate --holdout data/eval_holdout.jsonl
PYTHONPATH=src python -m auto_labeling.train_embed   # Embedding-Vergleich
# Merge-Experiment: train/evaluate mit --labels data/labels.merged.jsonl,
#   --out/--metrics auf *_merged zeigen lassen
```

## Wichtige Stolperfallen (hart erarbeitet)

1. **`search` hat `--limit-queries` Default 100.** Es sind erst 154 von 190
   Queries gesucht — **36 Queries fehlen noch** (die zuletzt ergänzten
   Research-Traps und einige deutsche Negatives). Nachziehen mit
   `--limit-queries 300`.
2. **`fetch` hat Default-Limit 100** pro Lauf. Für volle Runden explizit
   `--limit` setzen.
3. **Teacher-Fehlerklasse Sprache:** ~5 % der positives sind deutsche Seiten,
   die qwen2.5:7b fälschlich als `english=true` labelt (immowelt, wg-gesucht,
   tuebingen.de/Bürgerservice, eventbrite.de …). Gemischtsprachige Seiten
   (Stocherkahn-Touristik) werden inkonsistent gelabelt. Fix: deterministischer
   Sprachcheck (z. B. `langdetect`) über den Snapshot-Text VOR dem
   Teacher-Call; deutsche Seiten direkt negative.
4. **~430 Snapshots sind Fetch-Fehlschläge ohne Text** (v. a. HTTP 403
   Bot-Blocks von TripAdvisor/Michelin/Yelp, dazu 429, PDFs, Timeouts). Nicht
   labelbar, einfach ignorieren.
5. **`make_text` nutzt Titel + URL + Snippet + Seitentext (3000 Zeichen).**
   Der Text steckt direkt in `labels.final.jsonl`/`eval_holdout.jsonl`
   (aus den Snapshots gejoint) — wer die Dateien neu erzeugt, muss den
   Join wiederholen. `fetch` und `label` laufen parallel (`--workers`);
   für parallele Label-Läufe braucht der Ollama-Server
   `OLLAMA_NUM_PARALLEL >= workers` (gesetzt via launchctl, gilt bis Reboot).
6. **gray wird im Training bereits verworfen** (`load_examples` filtert auf
   positive/negative). Nur 15 gray-Labels insgesamt — keine Aktion nötig.
7. Serper-Key kommt aus Env oder `.env` (`SERPER_API_KEY`).

## Befehle

```bash
cd "/Users/julian/Documents/Projekte/Auto Labeling"
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1

python -m auto_labeling.cli search --resume --limit-queries 300
python -m auto_labeling.cli fetch  --resume --limit 1500  # parallel, --workers (Default 8)
python -m auto_labeling.cli label  --resume --limit 1500
python -m auto_labeling.cli make-review-batch --batch-size 150
# Review batches are loaded through the Teacher review mode in Labeling Lab.
python -m auto_labeling.cli train
python -m auto_labeling.cli evaluate
```

Fortschritts-Check während eines Label-Laufs:

```bash
python - <<'PY'
import json
from collections import Counter
from pathlib import Path
rows=[json.loads(l) for l in Path('data/teacher_labels.raw.jsonl').read_text().splitlines() if l.strip()]
print(len(rows), Counter(r.get('label') for r in rows))
PY
```

## Offene Schritte

Erledigt (10.07.): Datensammlung komplett, Teacher-Vergleich, Konflikt-Review,
Merge, Baseline + Embedding-Vergleich — siehe "Ergebnisse & Fazit".

Was noch aussteht:

1. **PageVerdict in die Suchmaschine integrieren** (Repo `search-engine`):
   Index-Filter beim Crawlen/Indexieren; danach Link-Modell, das aus den
   PageVerdict-Urteilen gecrawlter Seiten lernt (Frontier-Priorisierung).
2. **Optional, falls Qualität nicht reicht:** Sprachfilter (`langdetect`) als
   deterministischer Vorfilter; Host-basierter Train/Test-Split gegen
   Host-Leakage (uni-tuebingen.de dominiert); Schwellwert-Tuning auf die
   negative-Precision-Schwäche.
3. **Report-Notiz:** Modellwahl begründen — TF-IDF+LogReg selbst gebaut,
   Embedding-Variante getestet und verworfen (Zahlen oben), BGE-M3/E5 wegen
   FAQ 5 ausgeschlossen. Teacher-Distillation (LLM labelt, kleines Modell
   lernt) als Design-Entscheidung erklären.

## Crawl-Endstand (2026-07-11 morgens)

Der produktive Crawl mit dem Auto-Labeling-PageVerdict endete bei
**4.252 Accepts auf 157 Hosts** (Supervisor-Festfahr-Erkennung: Seed-Set
erschöpft; 118 Seeds inkl. Tunnel-, Kategorie- und Entitäts-Nachschub).
Kategorie-Verteilung: Uni 23 %, Transport 18 % (viel dünner Routen-Content),
Outdoor 12 %, Food 8,6 %, Accommodation 8,2 %, Events 6,8 %, Attractions 6,3 %
(+ ~12 % allgemeine Reiseguides), History nur 0,4 %.
Qualität (Teacher-Spot-Check bei ~1.700): 98 % englisch, 92 % Tübingen-Bezug.
Bekannte harte Grenze: tuebingen-info.de (offizielle Touristik) hat KEINE
crawlbare englische Version — /en liefert deutsches HTML.
Empfehlung: Mit diesem Index Richtung Ranking/Retrieval arbeiten; mehr
Crawling bringt fast nichts mehr.

## Seed-Nachschub (wiederholbarer Prozess)

Zwei erprobte Wege, dem Crawler neue Adern zu geben, beide ohne
Crawler-Änderung:

1. **Tunnel-Sweep** (`tools/tunnel_sweep.py`): fetcht die non_english-Rejects
   des Crawls erneut und extrahiert unbekannte `/en/`-Links (Ein-Hop-Tunneling
   als Batch). Kandidaten nach Tübingen-Bezug filtern, beste als Seeds in
   `search-engine/crawl/seeds.toml`. Idempotent — Bekanntes wird übersprungen,
   lohnt also nach jedem längeren Crawl-Lauf erneut. Fund vom 11.07.:
   Kunsthalle, Botanischer Garten, Naturpark Schönbuch, studIT u. a.
2. **Teacher-positive SERP-Hosts als Seeds** (bereits zweimal gemacht) bzw.
   Auto-SeedFinder: charakteristische Terme aus akzeptierten Seiten ->
   neue Serper-Queries -> unbekannte Hosts als Seeds. Noch nicht
   automatisiert; die manuelle Variante steckt in der Session-Historie.

Crawler-Neustart nach Seed-Änderung: einfach `pkill -f "uv run crawl"` — der
Supervisor (falls aktiv, s. `data/crawl_supervisor.log` im Auto-Labeling-Repo)
startet automatisch neu und lädt die Seeds frisch.

## Wovon die Finger lassen

- Kein BGE-M3/E5/nomic-embed o. ä. (Kurs-Regel, s. o.).
- Kein Fine-Tuning des Encoders — bei ~2000 Labels frozen + LogReg.
- Kein LLM-Klassifikator zur Laufzeit (zu langsam, und genau das wird ja in
  das Studentenmodell destilliert).
- Keine parallelen Pipeline-Läufe auf denselben JSONL-Dateien (ein
  `label`-Prozess zur Zeit; `--resume` macht Wiederanlauf billig).
