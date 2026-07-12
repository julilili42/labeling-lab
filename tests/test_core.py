from pathlib import Path
import tempfile
import unittest

from auto_labeling.fetcher import parse_html
from auto_labeling.cli import query_progress
from auto_labeling.jsonl import read_jsonl, write_jsonl
from auto_labeling.queries import parse_query_line
from auto_labeling.teacher import postprocess_label
from auto_labeling.train import make_text
from auto_labeling.urls import normalize_url


class CoreTests(unittest.TestCase):
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
            "<html><head><title>Hello</title><meta name='description' content='Desc'>"
            "<script>bad()</script></head><body><h1>Main</h1><h2>Sub</h2>Text</body></html>"
        )
        self.assertEqual(parsed["title"], "Hello")
        self.assertEqual(parsed["description"], "Desc")
        self.assertEqual(parsed["h1"], "Main")
        self.assertIn("Text", parsed["text"])
        self.assertNotIn("bad", parsed["text"])

    def test_postprocess_forces_research_negative(self):
        label = postprocess_label(
            {
                "english": True,
                "tuebingen_related": True,
                "general_search_useful": False,
                "research_only": True,
                "label": "positive",
                "confidence": 0.95,
                "reason": "research",
            }
        )
        self.assertEqual(label["label"], "negative")

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

    def test_training_text_matches_pageverdict_shape(self):
        text = make_text({"title": "A", "url": "https://example.com/x", "snippet": "B"})
        self.assertIn("title: A", text)
        self.assertIn("url: https://example.com/x", text)
        self.assertIn("display_url: example.com/x", text)
        self.assertIn("snippet: B", text)


if __name__ == "__main__":
    unittest.main()
