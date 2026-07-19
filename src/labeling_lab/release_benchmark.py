from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .dataset import file_hash, read_jsonl

BOOTSTRAP_SAMPLES = 1_000


def _classification_metrics(
    labels: list[str], scores: list[float], threshold: float
) -> dict[str, object]:
    actual = [label == "positive" for label in labels]
    predicted = [score >= threshold for score in scores]
    return {
        "examples": len(labels),
        "positives": sum(actual),
        "threshold": threshold,
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "matthews_correlation": float(matthews_corrcoef(actual, predicted)),
        "roc_auc": float(roc_auc_score(actual, scores)),
        "average_precision": float(average_precision_score(actual, scores)),
        "brier_score": float(brier_score_loss(actual, scores)),
        "log_loss": float(log_loss(actual, scores, labels=[False, True])),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[False, True]).tolist(),
    }


def _bootstrap_intervals(
    rows: list[dict[str, object]],
    scores: list[float],
    threshold: float,
    *,
    seed: int,
) -> dict[str, dict[str, float]]:
    indices_by_host: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices_by_host[str(row["group"])].append(index)
    hosts = sorted(indices_by_host)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_SAMPLES):
        indices = [
            index
            for _ in hosts
            for index in indices_by_host[rng.choice(hosts)]
        ]
        labels = [str(rows[index]["label"]) for index in indices]
        if len(set(labels)) != 2:
            continue
        metrics = _classification_metrics(
            labels, [scores[index] for index in indices], threshold
        )
        for name, value in metrics.items():
            if isinstance(value, float):
                samples[name].append(value)
    intervals = {}
    for name, values in samples.items():
        ordered = sorted(values)
        intervals[name] = {
            "lower": ordered[int(0.025 * (len(ordered) - 1))],
            "upper": ordered[int(0.975 * (len(ordered) - 1))],
        }
    return intervals


def _threshold_rows(task: str, labels: list[str], scores: list[float]):
    actual = [label == "positive" for label in labels]
    for step in range(101):
        threshold = step / 100
        predicted = [score >= threshold for score in scores]
        yield {
            "task": task,
            "threshold": threshold,
            "precision": precision_score(actual, predicted, zero_division=0),
            "recall": recall_score(actual, predicted, zero_division=0),
            "f1": f1_score(actual, predicted, zero_division=0),
            "predicted_positive": sum(predicted),
        }


def _calibration_rows(task: str, labels: list[str], scores: list[float]):
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    for bin_index in range(10):
        start = len(ordered) * bin_index // 10
        end = len(ordered) * (bin_index + 1) // 10
        values = ordered[start:end]
        if values:
            yield {
                "task": task,
                "bin": bin_index,
                "examples": len(values),
                "mean_score": sum(score for score, _ in values) / len(values),
                "positive_rate": sum(label == "positive" for _, label in values) / len(values),
            }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def benchmark_release(release_dir: Path) -> dict[str, object]:
    manifest_path = release_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seed = int(manifest["random_seed"])
    reports: dict[str, object] = {}
    predictions: list[dict[str, object]] = []
    thresholds: list[dict[str, object]] = []
    calibration: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for task in ("page", "link"):
        dataset_path = release_dir / "datasets" / task / "test.jsonl"
        model_path = release_dir / "models" / f"{task}_verdict.joblib"
        expected_hash = manifest["tasks"][task]["split_sha256"]["test"]
        if file_hash(dataset_path) != expected_hash:
            raise ValueError(f"{task} test split differs from the release manifest")
        rows = list(read_jsonl(dataset_path))
        bundle = joblib.load(model_path)
        model = bundle["model"]
        positive = list(model.classes_).index("positive")
        scores = [
            float(value)
            for value in model.predict_proba([str(row["text"]) for row in rows])[:, positive]
        ]
        labels = [str(row["label"]) for row in rows]
        threshold = float(bundle["positive_threshold"])
        metrics = _classification_metrics(labels, scores, threshold)
        intervals = _bootstrap_intervals(rows, scores, threshold, seed=seed)
        reports[task] = {
            "test_dataset_sha256": file_hash(dataset_path),
            "model_sha256": file_hash(model_path),
            "groups": len({row["group"] for row in rows}),
            "metrics": metrics,
            "confidence_intervals_95": intervals,
        }
        metric_rows.extend(
            {
                "task": task,
                "metric": name,
                "value": value,
                "lower_95": intervals.get(name, {}).get("lower", ""),
                "upper_95": intervals.get(name, {}).get("upper", ""),
            }
            for name, value in metrics.items()
            if isinstance(value, (int, float))
        )
        predictions.extend(
            {
                "task": task,
                "id": row["id"],
                "url": row["url"],
                "host": row["group"],
                "label": row["label"],
                "score": score,
                "prediction": "positive" if score >= threshold else "negative",
                "source": row["source"],
                "stratum": row.get("stratum", ""),
            }
            for row, score in zip(rows, scores, strict=True)
        )
        thresholds.extend(_threshold_rows(task, labels, scores))
        calibration.extend(_calibration_rows(task, labels, scores))

    benchmark_dir = release_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    (benchmark_dir / "report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_manifest_sha256": file_hash(manifest_path),
                "bootstrap_unit": "host",
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "tasks": reports,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(benchmark_dir / "metrics.csv", metric_rows)
    _write_csv(benchmark_dir / "predictions.csv", predictions)
    _write_csv(benchmark_dir / "thresholds.csv", thresholds)
    _write_csv(benchmark_dir / "calibration.csv", calibration)
    return reports
