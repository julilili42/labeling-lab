from pathlib import Path
import tempfile
import unittest

from auto_labeling.fetcher import parse_html
from auto_labeling.cli import _resume_rows, cmd_apply_reviews, query_progress
from auto_labeling.jsonl import read_jsonl, write_jsonl
from auto_labeling.queries import parse_query_line
from auto_labeling.teacher import PAGE_PROMPT_HASH, label_snapshot, postprocess_label
from auto_labeling.urls import normalize_url
from labeling_lab.dataset import page_text


class CoreTests(unittest.TestCase):
    def test_page_prompt_hash_is_frozen(self):
        self.assertEqual(
            PAGE_PROMPT_HASH,
            "b91f9d2e32a8d1a91170d15be35ba14099d2909ba9621f91ec0eaf3ece360c9a",
        )

    def test_parse_query_line_with_count(self):
        spec = parse_query_line("Tuebingen tourism | 50")
        self.assertEqual(spec.query, "Tuebingen tourism")
        self.assertEqual(spec.results, 50)

    def test_normalize_url_removes_tracking(self):
        self.assertEqual(
            normalize_url("https://www.Example.com/path/?utm_source=x&b=2&a=1#top"),
            "https://example.com/path?a=1&b=2",
        )

    def test_parse_html_extracts_visible_fields(self):
        parsed = parse_html(
            "<html lang='de'><head><title>Hello</title><meta name='description' content='Desc'>"
            "<script>bad()</script></head><body><h1>Main</h1><h2>Sub</h2>Text</body></html>"
        )
        self.assertEqual(parsed["title"], "Hello")
        self.assertEqual(parsed["description"], "Desc")
        self.assertEqual(parsed["h1"], "Main")
        self.assertEqual(parsed["html_lang"], "de")
        self.assertIn("Text", parsed["text"])
        self.assertNotIn("bad", parsed["text"])

    def test_postprocess_forces_research_negative_even_when_teacher_calls_it_useful(self):
        label = postprocess_label(
            {
                "english": True,
                "tuebingen_related": True,
                "general_search_useful": True,
                "research_only": True,
                "label": "positive",
                "confidence": 0.95,
                "reason": "research",
            }
        )
        self.assertEqual(label["label"], "negative")

    def test_postprocess_rejects_clearly_german_destination(self):
        label = postprocess_label(
            {
                "english": True,
                "tuebingen_related": True,
                "general_search_useful": True,
                "research_only": False,
                "label": "positive",
                "confidence": 0.95,
                "title": "Hotels in Tübingen ab € 69/Nacht - Auf KAYAK suchen",
            }
        )
        self.assertFalse(label["english"])
        self.assertEqual(label["label"], "negative")

    def test_fresh_teacher_label_overrides_stale_snapshot_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            label = label_snapshot(
                {
                    "id": "one",
                    "url": "https://example.test/tuebingen",
                    "text": "Useful Tuebingen visitor guide",
                    "text_hash": "one",
                    "label": "negative",
                    "english": False,
                },
                teacher="mock",
                model="mock",
                cache_dir=Path(tmp),
            )
            self.assertEqual(label["label"], "positive")

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            write_jsonl(path, [{"a": 1}, {"b": 2}])
            self.assertEqual(list(read_jsonl(path)), [{"a": 1}, {"b": 2}])

    def test_query_progress_uses_highest_rank_per_query(self):
        progress = query_progress(
            [
                {"query": "A", "rank": 10},
                {"query": "A", "rank": 5},
                {"query": "B", "rank": 2},
            ]
        )
        self.assertEqual(progress, {"A": 10, "B": 2})

    def test_resume_rejects_mixed_prompt_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "prompt_version": "new",
                        "prompt_sha256": "old-hash",
                        "teacher": "ollama",
                        "model": "qwen",
                    }
                ],
            )
            with self.assertRaises(SystemExit):
                _resume_rows(
                    path,
                    resume=True,
                    refresh=False,
                    prompt_version="new",
                    prompt_hash="new-hash",
                    teacher="ollama",
                    model="qwen",
                )

    def test_training_text_matches_pageverdict_shape(self):
        text = page_text(
            {
                "title": "A",
                "url": "https://example.com/x",
                "display_url": "example.com/x",
                "snippet": "B",
            }
        )
        self.assertIn("title: A", text)
        self.assertIn("url: https://example.com/x", text)
        self.assertIn("display_url: example.com/x", text)
        self.assertIn("snippet: B", text)

    def test_apply_reviews_updates_matching_final_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels.jsonl"
            reviewed = root / "reviewed.jsonl"
            write_jsonl(labels, [{"text_hash": "one", "label": "positive"}])
            write_jsonl(reviewed, [{"text_hash": "one", "label": "negative", "rating": 1}])
            empty = root / "empty.jsonl"
            write_jsonl(empty, [])
            cmd_apply_reviews(
                type(
                    "Args",
                    (),
                    {"labels": labels, "reviewed": reviewed, "teacher": empty, "snapshots": empty, "holdout": empty},
                )()
            )
            self.assertEqual(list(read_jsonl(labels))[0]["label"], "negative")


if __name__ == "__main__":
    unittest.main()
