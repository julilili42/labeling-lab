from labeling_lab.metrics import label_counts, select_threshold, threshold_metrics


def test_label_counts_counts_labels():
    assert label_counts(["positive", "negative", "positive"]) == {
        "positive": 2,
        "negative": 1,
    }


def test_threshold_metrics_reports_positive_class_stats():
    metrics = threshold_metrics(
        ["negative", "positive", "positive"],
        [0.2, 0.6, 0.9],
        thresholds=[0.5],
    )

    assert metrics["0.50"]["positive_support"] == 2
    assert metrics["0.50"]["predicted_positive"] == 2
    assert metrics["0.50"]["positive_precision"] == 1.0
    assert metrics["0.50"]["positive_recall"] == 1.0


def test_select_threshold_prefers_high_f1_among_precise_candidates():
    thresholds = {
        "0.50": {
            "positive_precision": 0.75,
            "positive_recall": 1.0,
            "positive_f1": 0.86,
        },
        "0.70": {
            "positive_precision": 0.85,
            "positive_recall": 0.7,
            "positive_f1": 0.77,
        },
        "0.80": {
            "positive_precision": 0.9,
            "positive_recall": 0.8,
            "positive_f1": 0.85,
        },
    }

    assert select_threshold(thresholds, min_precision=0.8) == 0.8
