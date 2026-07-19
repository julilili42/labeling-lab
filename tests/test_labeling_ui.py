import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient
import labeling_lab.server as server
from labeling_lab.server import (
    PipelineRequest,
    _pipeline_command,
    init_schema,
    link_candidates,
    read_crawler_candidates,
    read_link_candidates,
    review_item,
    upsert_link_results,
)


def make_connection() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    init_schema(con)
    return con


def link_record(host: str, index: int, raw_score: float = 9.0) -> dict[str, object]:
    target_url = f"https://{host}/page-{index}"
    return {
        "parent_url": f"https://parent.example/{host}/{index}",
        "parent_host": "parent.example",
        "parent_depth": 0,
        "parent_pageverdict_score": 0.9,
        "parent_pageverdict_label": "positive",
        "parent_pageverdict_decision": "index_strong",
        "parent_relevance": 8.0,
        "anchor": f"Tuebingen link {index}",
        "target_url": target_url,
        "target_host": host,
        "target_depth": 1,
        "raw_score": raw_score,
        "should_enqueue": True,
        "selected": True,
        "rejection_reason": None,
        "target_status": None,
        "target_status_code": None,
        "target_content_type": None,
        "target_language": None,
        "target_relevance": None,
        "target_token_count": None,
        "target_pageverdict_score": 0.52,
        "target_pageverdict_label": "positive",
        "target_pageverdict_decision": "index_cautious",
        "target_exclusion_reason": None,
        "target_fetched_at": None,
        "source": "crawler_linkverdict",
        "notes": "",
    }


def test_link_candidates_round_robins_hosts_before_repeating_a_dominant_host():
    con = make_connection()
    dominant = [link_record("dominant.example", index) for index in range(10)]
    diverse = [
        link_record("b.example", 1, raw_score=8.0),
        link_record("c.example", 1, raw_score=7.0),
        link_record("d.example", 1, raw_score=6.0),
    ]
    with con:
        upsert_link_results(con, [*dominant, *diverse])

    results = link_candidates(con, limit=4, unlabeled_only=True)

    assert len(results) == 4
    assert len({result.target_host for result in results}) == 4


def test_link_candidates_push_navigation_noise_behind_content_links():
    con = make_connection()
    navigation = link_record("nav.example", 1, raw_score=10.0)
    navigation["parent_url"] = navigation["target_url"]
    navigation["anchor"] = "Jump to content"
    navigation["rejection_reason"] = "seen_url"

    content = link_record("content.example", 1, raw_score=5.0)
    content["anchor"] = "Tuebingen city history"

    with con:
        upsert_link_results(con, [navigation, content])

    [first] = link_candidates(con, limit=1, unlabeled_only=True)

    assert first.target_host == "content.example"


def test_reads_page_candidates_from_jsonl(tmp_path):
    path = tmp_path / "pageverdict_candidates.jsonl"
    path.write_text(
        json.dumps(
            {
                "query": "crawler:index_strong",
                "rank": 1,
                "title": "Tuebingen",
                "url": "https://example.com/",
                "display_url": "example.com",
                "snippet": "A useful page.",
                "source": "crawler_pageverdict",
                "pageverdict_score": 0.91,
                "pageverdict_label": "positive",
                "pageverdict_decision": "index_strong",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    [result] = read_crawler_candidates(path, "crawler:pageverdict")

    assert result["title"] == "Tuebingen"
    assert result["pageverdict_score"] == 0.91
    assert result["pageverdict_decision"] == "index_strong"


def test_reads_link_candidates_from_jsonl(tmp_path):
    path = tmp_path / "link_candidates.jsonl"
    record = link_record("example.com", 1)
    record["target_status"] = "page"
    record["target_pageverdict_score"] = 0.88
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    [result] = read_link_candidates(path)

    assert result["target_url"] == "https://example.com/page-1"
    assert result["source"] == "crawler_linkverdict"
    assert result["target_status"] == "page"
    assert result["target_pageverdict_score"] == 0.88


def test_teacher_review_item_uses_snapshot_text_when_no_snippet():
    item = review_item(
        {
            "title": "Visit",
            "url": "https://example.test/visit",
            "text": "Visible page text",
            "label": "positive",
            "teacher": "ollama",
        }
    )

    assert item["snippet"] == "Visible page text"
    assert item["source"] == "teacher_review"
    assert item["rating"] is None


def test_ui_imports_and_rates_teacher_review_batches():
    script = (Path(__file__).parents[1] / "static/app.js").read_text()

    assert "/api/import/review-batch" in script
    assert "/api/review-rating" in script


def test_review_batch_import_resumes_after_existing_ratings(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    batch = tmp_path / "batch.jsonl"
    batch.write_text('{"id":"done"}\n{"id":"pending"}\n', encoding="utf-8")
    (tmp_path / "data" / "labels.reviewed.jsonl").write_text(
        '{"id":"done","label":"positive"}\n', encoding="utf-8"
    )
    monkeypatch.setattr(server, "ROOT_DIR", tmp_path)

    response = TestClient(server.app).post(
        "/api/import/review-batch", json={"path": "batch.jsonl"}
    )

    assert response.status_code == 200
    assert [row["id"] for row in response.json()["results"]] == ["pending"]
