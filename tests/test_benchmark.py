import hashlib
import json

import joblib
from sklearn.dummy import DummyClassifier

from labeling_lab.benchmark import create_review_queue, score


def model(path):
    classifier = DummyClassifier(strategy="prior")
    classifier.fit(["positive", "negative"], ["positive", "negative"])
    joblib.dump({"model": classifier}, path)


def test_replay_scores_are_deterministic(tmp_path):
    artifact = tmp_path / "model.joblib"
    model(artifact)
    rows = [{"kind": "page", "url": "https://example.test", "title": "Tuebingen", "text": "Visit"}]
    assert score(artifact, rows, kind="page") == score(artifact, rows, kind="page")


def test_review_queue_is_stratified_and_excludes_training_text(tmp_path):
    snapshot = tmp_path / "features.jsonl"
    rows = [
        {"kind": "page", "url": "https://train.test", "text": "Training text", "outcome": "accepted"},
        {"kind": "page", "url": "https://good.test", "text": "Good text", "outcome": "accepted"},
        {"kind": "page", "url": "https://edge.test", "text": "Edge text", "outcome": "rejected", "score": .61},
        {"kind": "link", "target_url": "https://high.test", "linkverdict_score": .9},
        {"kind": "link", "target_url": "https://low.test", "linkverdict_score": .1},
    ]
    snapshot.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"text_hash": hashlib.sha256("training text".encode()).hexdigest()}) + "\n", encoding="utf-8")

    queue = create_review_queue(snapshot, tmp_path / "queue.jsonl", exclude=[labels], per_stratum=1)

    assert {entry["stratum"] for entry in queue} == {"accepted_page", "borderline_page", "high_link", "low_link"}
    assert all(entry["item"].get("url") != "https://train.test" for entry in queue)
