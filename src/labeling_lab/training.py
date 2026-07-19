from __future__ import annotations

import json
import statistics
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from .dataset import file_hash, read_jsonl, stable_hash


def build_features() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
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


def build_classifier(*, c: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=c,
        class_weight="balanced",
        max_iter=3000,
        random_state=seed,
    )


def build_model(*, c: float, seed: int) -> Pipeline:
    return Pipeline(
        [("features", build_features()), ("classifier", build_classifier(c=c, seed=seed))]
    )


def _scores(model: Pipeline, rows: list[dict[str, object]]) -> list[float]:
    positive = list(model.classes_).index("positive")
    texts = [str(row["text"]) for row in rows]
    return [float(value) for value in model.predict_proba(texts)[:, positive]]


def _threshold(labels: list[str], scores: list[float]) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores, pos_label="positive")
    candidates = []
    for index, threshold in enumerate(thresholds):
        denominator = precision[index] + recall[index]
        f1 = 0.0 if denominator == 0 else 2 * precision[index] * recall[index] / denominator
        candidates.append((f1, precision[index], float(threshold)))
    return max(candidates)[2]


def train_task(
    dataset_dir: Path,
    model_path: Path,
    *,
    seed: int,
    positive_threshold: float | None = None,
) -> dict[str, object]:
    train_path = dataset_dir / "train.jsonl"
    validation_path = dataset_dir / "validation.jsonl"
    train = list(read_jsonl(train_path))
    validation = list(read_jsonl(validation_path))
    train_labels = [str(row["label"]) for row in train]
    validation_labels = [str(row["label"]) for row in validation]
    train_texts = [str(row["text"]) for row in train]
    train_groups = [str(row["group"]) for row in train]
    c_values = (0.5, 1.0, 2.0, 4.0)
    scores_by_c: dict[float, list[float]] = {c: [] for c in c_values}
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fit_indices, score_indices in splitter.split(train_texts, train_labels, train_groups):
        features = build_features()
        fit_x = features.fit_transform([train_texts[index] for index in fit_indices])
        score_x = features.transform([train_texts[index] for index in score_indices])
        fit_y = [train_labels[index] for index in fit_indices]
        score_y = [train_labels[index] for index in score_indices]
        for c in c_values:
            classifier = build_classifier(c=c, seed=seed)
            classifier.fit(fit_x, fit_y)
            positive = list(classifier.classes_).index("positive")
            scores = classifier.predict_proba(score_x)[:, positive]
            scores_by_c[c].append(
                float(average_precision_score(score_y, scores, pos_label="positive"))
            )
    candidates: list[dict[str, object]] = []
    for c, fold_scores in scores_by_c.items():
        candidates.append(
            {
                "c": c,
                "cv_average_precision_mean": statistics.mean(fold_scores),
                "cv_average_precision_std": statistics.pstdev(fold_scores),
            }
        )
    selected = max(candidates, key=lambda item: item["cv_average_precision_mean"])
    validation_model = build_model(c=float(selected["c"]), seed=seed)
    validation_model.fit(train_texts, train_labels)
    validation_scores = _scores(validation_model, validation)
    validation_threshold = _threshold(validation_labels, validation_scores)
    threshold = positive_threshold or validation_threshold
    task = str(train[0]["task"])
    development_hash = stable_hash(
        {"train": file_hash(train_path), "validation": file_hash(validation_path)}
    )
    bundle = {
        "model": validation_model,
        "labels": ["negative", "positive"],
        "positive_threshold": threshold,
        "training_examples": len(train),
        "development_dataset_sha256": development_hash,
        "feature_contract": f"{task}-verdict-v1",
        "feature_fields": (
            ["title", "url", "display_url", "snippet", "text"]
            if task == "page"
            else [
                "anchor",
                "target_url",
                "target_host",
                "parent_url",
                "parent_host",
                "parent_depth",
                "parent_pageverdict_score",
                "parent_pageverdict_decision",
                "parent_relevance",
                "target_depth",
            ]
        ),
        "selected_c": selected["c"],
        "random_seed": seed,
    }
    if positive_threshold is not None:
        bundle["validation_f1_threshold"] = validation_threshold
        bundle["threshold_source"] = "configured_development_set"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    report = {
        "task": task,
        "training_examples": len(train),
        "validation_examples": len(validation),
        "selected_c": selected["c"],
        "positive_threshold": threshold,
        "validation_average_precision": float(
            average_precision_score(validation_labels, validation_scores, pos_label="positive")
        ),
        "validation_roc_auc": float(roc_auc_score(validation_labels, validation_scores)),
        "candidates": candidates,
        "model_sha256": file_hash(model_path),
    }
    if positive_threshold is not None:
        report["validation_f1_threshold"] = validation_threshold
        report["threshold_source"] = "configured_development_set"
    return report


def train_release(release_dir: Path) -> dict[str, object]:
    manifest = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
    seed = int(manifest["random_seed"])
    for task in ("page", "link"):
        for split in ("train", "validation"):
            path = release_dir / "datasets" / task / f"{split}.jsonl"
            expected = manifest["tasks"][task]["split_sha256"][split]
            if file_hash(path) != expected:
                raise ValueError(f"{task} {split} split differs from the release manifest")
    reports = {
        task: train_task(
            release_dir / "datasets" / task,
            release_dir / "models" / f"{task}_verdict.joblib",
            seed=seed,
            positive_threshold=manifest["tasks"][task].get("positive_threshold"),
        )
        for task in ("page", "link")
    }
    path = release_dir / "training.json"
    path.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return reports
