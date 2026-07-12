from __future__ import annotations

from typing import Sequence

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

LABELS = ["negative", "positive"]


def label_counts(labels: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


def group_split(
    labels: Sequence[str],
    groups: Sequence[str],
    *,
    test_size: float,
    random_state: int,
    min_groups: int = 8,
) -> tuple[list[int], list[int], str]:
    indices = list(range(len(labels)))
    if len(set(groups)) >= min_groups:
        for offset in range(100):
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state + offset
            )
            train_idx, valid_idx = next(splitter.split(indices, labels, groups))
            train_labels = {labels[i] for i in train_idx}
            valid_labels = {labels[i] for i in valid_idx}
            if len(train_labels) == 2 and len(valid_labels) == 2:
                return list(train_idx), list(valid_idx), "group_shuffle"

    train_idx, valid_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=list(labels)
    )
    return list(train_idx), list(valid_idx), "stratified_random"


def evaluate(model, texts, labels, *, with_thresholds: bool = False) -> dict[str, object]:
    y_pred = model.predict(texts).tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, y_pred, labels=LABELS, zero_division=0
    )
    metrics: dict[str, object] = {
        "accuracy": accuracy_score(labels, y_pred),
        "labels": {
            label: {
                "precision": precision[index],
                "recall": recall[index],
                "f1": f1[index],
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": confusion_matrix(labels, y_pred, labels=LABELS).tolist(),
        },
        "classification_report": classification_report(
            labels, y_pred, labels=LABELS, output_dict=True, zero_division=0
        ),
    }
    if hasattr(model, "predict_proba") and len(set(labels)) == 2:
        positive_index = list(model.classes_).index("positive")
        probabilities = model.predict_proba(texts)[:, positive_index].tolist()
        y_bin = [1 if label == "positive" else 0 for label in labels]
        metrics["roc_auc"] = roc_auc_score(y_bin, probabilities)
        metrics["pr_auc"] = average_precision_score(y_bin, probabilities)
        if with_thresholds:
            metrics["thresholds"] = threshold_metrics(labels, probabilities)
    return metrics


def threshold_metrics(
    y_true: Sequence[str],
    probabilities: Sequence[float],
    thresholds: Sequence[float] | None = None,
) -> dict[str, dict[str, float | int]]:
    values: dict[str, dict[str, float | int]] = {}
    for threshold in thresholds or [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y_pred = ["positive" if probability >= threshold else "negative" for probability in probabilities]
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, zero_division=0
        )
        values[f"{threshold:.2f}"] = {
            "positive_precision": precision[1],
            "positive_recall": recall[1],
            "positive_f1": f1[1],
            "positive_support": int(support[1]),
            "predicted_positive": y_pred.count("positive"),
        }
    return values


def select_threshold(
    thresholds: dict[str, dict[str, float | int]],
    *,
    min_precision: float = 0.8,
    fallback_min_threshold: float = 0.6,
    fallback_min_recall: float = 0.5,
) -> float:
    candidates = [
        (float(threshold), values)
        for threshold, values in thresholds.items()
        if float(values["positive_precision"]) >= min_precision
    ]
    if not candidates:
        candidates = [
            (float(threshold), values)
            for threshold, values in thresholds.items()
            if float(threshold) >= fallback_min_threshold
            and float(values["positive_recall"]) >= fallback_min_recall
        ]
    if not candidates:
        candidates = [
            (float(threshold), values)
            for threshold, values in thresholds.items()
            if float(threshold) >= fallback_min_threshold
        ]
    if not candidates:
        candidates = [(float(threshold), values) for threshold, values in thresholds.items()]
    if all(float(values["positive_precision"]) < min_precision for _, values in candidates):
        best_threshold, _ = max(
            candidates,
            key=lambda item: (
                float(item[1]["positive_precision"]),
                float(item[1]["positive_f1"]),
                float(item[1]["positive_recall"]),
                item[0],
            ),
        )
        return best_threshold

    best_threshold, _ = max(
        candidates,
        key=lambda item: (
            float(item[1]["positive_f1"]),
            float(item[1]["positive_recall"]),
            item[0],
        ),
    )
    return best_threshold
