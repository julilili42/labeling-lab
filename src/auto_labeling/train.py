from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

from .jsonl import read_jsonl


def normalize_space(value: object) -> str:
    return " ".join(str(value or "").split())


def display_url(url: str) -> str:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").removeprefix("www.")
    path = parsed.path.strip("/")
    return f"{host}/{path}".strip("/")


def make_text(row: dict[str, object]) -> str:
    url = normalize_space(row.get("url") or row.get("normalized_url"))
    parts = [
        f"title: {normalize_space(row.get('title'))}",
        f"url: {url}",
        f"display_url: {normalize_space(row.get('display_url') or display_url(url))}",
        f"snippet: {normalize_space(row.get('snippet'))}",
        f"text: {normalize_space(row.get('text'))[:3000]}",
    ]
    return "\n".join(parts)


def load_examples(path: Path) -> list[dict[str, object]]:
    examples = []
    for row in read_jsonl(path):
        label = str(row.get("label") or "")
        if label not in {"positive", "negative"}:
            continue
        examples.append({**row, "text": make_text(row)})
    return examples


def _imports():
    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import FeatureUnion, Pipeline
    except ImportError as exc:
        raise SystemExit("Training needs optional dependencies: pip install -e '.[train]'") from exc
    return {
        "joblib": joblib,
        "TfidfVectorizer": TfidfVectorizer,
        "LogisticRegression": LogisticRegression,
        "accuracy_score": accuracy_score,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "train_test_split": train_test_split,
        "FeatureUnion": FeatureUnion,
        "Pipeline": Pipeline,
    }


def build_model():
    sk = _imports()
    text_features = sk["FeatureUnion"](
        [
            (
                "word",
                sk["TfidfVectorizer"](
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                sk["TfidfVectorizer"](
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
        ]
    )
    return sk["Pipeline"](
        [
            ("features", text_features),
            (
                "classifier",
                sk["LogisticRegression"](
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=13,
                ),
            ),
        ]
    )


def _label_counts(labels: list[str]) -> dict[str, int]:
    return {label: labels.count(label) for label in sorted(set(labels))}


def train_model(labels_path: Path, out_path: Path, *, metrics_path: Path) -> dict[str, object]:
    sk = _imports()
    examples = load_examples(labels_path)
    labels = [str(example["label"]) for example in examples]
    if len(examples) < 20 or len(set(labels)) < 2:
        raise SystemExit("Need at least 20 positive/negative examples with both classes")

    texts = [str(example["text"]) for example in examples]
    train_x, valid_x, train_y, valid_y = sk["train_test_split"](
        texts, labels, test_size=0.2, random_state=13, stratify=labels
    )
    model = build_model()
    model.fit(train_x, train_y)
    predictions = model.predict(valid_x)

    final_model = build_model()
    final_model.fit(texts, labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sk["joblib"].dump(
        {
            "model": final_model,
            "feature_fields": ["title", "url", "display_url", "snippet"],
            "labels": ["negative", "positive"],
            "ignored_rating": 3,
            "training_examples": len(examples),
            "source": "auto-labeling",
        },
        out_path,
    )
    report = {
        "examples": len(examples),
        "label_counts": _label_counts(labels),
        "validation_examples": len(valid_y),
        "accuracy": float(sk["accuracy_score"](valid_y, predictions)),
        "confusion_matrix": sk["confusion_matrix"](valid_y, predictions, labels=["negative", "positive"]).tolist(),
        "classification_report": sk["classification_report"](
            valid_y, predictions, labels=["negative", "positive"], output_dict=True, zero_division=0
        ),
        "model_path": str(out_path),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate_model(model_path: Path, holdout_path: Path, *, metrics_path: Path) -> dict[str, object]:
    sk = _imports()
    bundle = sk["joblib"].load(model_path)
    model = bundle["model"]
    examples = load_examples(holdout_path)
    if not examples:
        raise SystemExit(f"No positive/negative holdout examples in {holdout_path}")
    labels = [str(example["label"]) for example in examples]
    predictions = model.predict([str(example["text"]) for example in examples])
    report = {
        "examples": len(examples),
        "label_counts": _label_counts(labels),
        "accuracy": float(sk["accuracy_score"](labels, predictions)),
        "confusion_matrix": sk["confusion_matrix"](labels, predictions, labels=["negative", "positive"]).tolist(),
        "classification_report": sk["classification_report"](
            labels, predictions, labels=["negative", "positive"], output_dict=True, zero_division=0
        ),
        "model_path": str(model_path),
        "holdout_path": str(holdout_path),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
