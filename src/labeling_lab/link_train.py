from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from .link_features import LinkVerdictInput, host_from_url, make_text
from .metrics import evaluate, group_split, label_counts, select_threshold
from .modeling import build_tfidf_logreg, save_bundle

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data" / "labeling.sqlite"
DEFAULT_OUT = ROOT / "data" / "models"
DEFAULT_MODEL_NAME = "link_verdict.joblib"


@dataclass(frozen=True)
class Example:
    id: int
    parent_url: str
    parent_host: str
    parent_depth: int | None
    parent_pageverdict_score: float | None
    parent_pageverdict_decision: str
    parent_relevance: float | None
    anchor: str
    target_url: str
    target_host: str
    target_depth: int | None
    raw_score: float | None
    rating: int
    label: str
    text: str


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _input_from_row(row: sqlite3.Row) -> LinkVerdictInput:
    return LinkVerdictInput(
        anchor=row["anchor"],
        target_url=row["target_url"],
        parent_url=row["parent_url"],
        parent_host=row["parent_host"],
        parent_depth=_optional_int(row["parent_depth"]),
        parent_pageverdict_score=_optional_float(row["parent_pageverdict_score"]),
        parent_pageverdict_decision=row["parent_pageverdict_decision"] or "",
        parent_relevance=_optional_float(row["parent_relevance"]),
        target_host=row["target_host"],
        target_depth=_optional_int(row["target_depth"]),
        raw_score=_optional_float(row["raw_score"]),
    )


def load_examples(db_path: Path) -> list[Example]:
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT
                id,
                parent_url,
                parent_host,
                parent_depth,
                parent_pageverdict_score,
                parent_pageverdict_decision,
                parent_relevance,
                anchor,
                target_url,
                target_host,
                target_depth,
                raw_score,
                rating,
                label
            FROM link_results
            WHERE rating IS NOT NULL
                AND rating != 3
                AND label IN ('negative', 'positive')
            ORDER BY id
            """
        ).fetchall()

    examples: list[Example] = []
    for row in rows:
        model_input = _input_from_row(row)
        target_host = row["target_host"] or host_from_url(row["target_url"])
        examples.append(
            Example(
                id=int(row["id"]),
                parent_url=row["parent_url"] or "",
                parent_host=row["parent_host"] or "",
                parent_depth=_optional_int(row["parent_depth"]),
                parent_pageverdict_score=_optional_float(row["parent_pageverdict_score"]),
                parent_pageverdict_decision=row["parent_pageverdict_decision"] or "",
                parent_relevance=_optional_float(row["parent_relevance"]),
                anchor=row["anchor"] or "",
                target_url=row["target_url"] or "",
                target_host=target_host,
                target_depth=_optional_int(row["target_depth"]),
                raw_score=_optional_float(row["raw_score"]),
                rating=int(row["rating"]),
                label=row["label"],
                text=make_text(model_input),
            )
        )
    return examples


def load_outcome_examples(
    db_path: Path, *, max_per_target: int = 3, max_examples: int = 50_000
) -> list[Example]:
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT id, parent_url, parent_host, parent_depth,
                   parent_pageverdict_score, parent_pageverdict_decision,
                   parent_relevance, anchor, target_url, target_host,
                   target_depth, raw_score, target_status
            FROM link_candidates
            WHERE target_status = 'page'
               OR (
                    target_status = 'rejected'
                    -- duplicate_text rejects pointed at content the page model
                    -- liked; labeling them negative would teach the ranker to
                    -- avoid good links
                    AND COALESCE(target_exclusion_reason, '') != 'duplicate_text'
               )
            ORDER BY id
            LIMIT ?
            """,
            (max_examples,),
        ).fetchall()

    target_label: dict[str, str] = {}
    for row in rows:
        if row["target_status"] == "page":
            target_label[row["target_url"]] = "positive"
        else:
            target_label.setdefault(row["target_url"], "negative")

    per_target: dict[str, int] = {}
    examples: list[Example] = []
    for row in rows:
        target = row["target_url"]
        if per_target.get(target, 0) >= max_per_target:
            continue
        per_target[target] = per_target.get(target, 0) + 1
        label = target_label[target]
        model_input = _input_from_row(row)
        target_host = row["target_host"] or host_from_url(row["target_url"])
        examples.append(
            Example(
                id=-int(row["id"]),
                parent_url=row["parent_url"] or "",
                parent_host=row["parent_host"] or "",
                parent_depth=_optional_int(row["parent_depth"]),
                parent_pageverdict_score=_optional_float(row["parent_pageverdict_score"]),
                parent_pageverdict_decision=row["parent_pageverdict_decision"] or "",
                parent_relevance=_optional_float(row["parent_relevance"]),
                anchor=row["anchor"] or "",
                target_url=row["target_url"] or "",
                target_host=target_host,
                target_depth=_optional_int(row["target_depth"]),
                raw_score=_optional_float(row["raw_score"]),
                rating=5 if label == "positive" else 1,
                label=label,
                text=make_text(model_input),
            )
        )
    return examples


def build_model():
    return build_tfidf_logreg(
        word_ngram_range=(1, 3),
        char_ngram_range=(3, 6),
        max_iter=3000,
    )


def host_label_counts(examples: list[Example], label: str) -> int:
    return len({example.target_host for example in examples if example.label == label})


def write_validation_predictions(
    model, valid: list[Example], output_path: Path, positive_threshold: float
) -> None:
    x_valid = [example.text for example in valid]
    positive_index = list(model.classes_).index("positive")
    probabilities = model.predict_proba(x_valid)[:, positive_index]
    y_pred = [
        "positive" if probability >= positive_threshold else "negative"
        for probability in probabilities
    ]

    with output_path.open("w", encoding="utf-8") as file:
        for example, prediction, probability in zip(valid, y_pred, probabilities, strict=True):
            row = asdict(example)
            record = {
                "id": row["id"],
                "rating": row["rating"],
                "label": row["label"],
                "prediction": prediction,
                "positive_probability": round(float(probability), 6),
                "parent_host": row["parent_host"],
                "target_host": row["target_host"],
                "anchor": row["anchor"],
                "target_url": row["target_url"],
                "parent_url": row["parent_url"],
                "raw_score": row["raw_score"],
                "parent_pageverdict_score": row["parent_pageverdict_score"],
                "parent_pageverdict_decision": row["parent_pageverdict_decision"],
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the LinkVerdict classifier")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--crawl-db",
        type=Path,
        action="append",
        default=None,
        help="crawl pages.sqlite to mine outcome labels from (repeatable)",
    )
    parser.add_argument("--max-per-target", type=int, default=3)
    # ponytail: bounded crawl sample; add deterministic stratified sampling if metrics show host bias.
    parser.add_argument("--max-outcome-examples", type=int, default=50_000)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=13)
    args = parser.parse_args(argv)

    examples = load_examples(args.db)
    hand_count = len(examples)
    outcome_count = 0
    for crawl_db in args.crawl_db or []:
        outcome = load_outcome_examples(
            crawl_db,
            max_per_target=args.max_per_target,
            max_examples=args.max_outcome_examples,
        )
        outcome_count += len(outcome)
        examples.extend(outcome)

    seen: set[tuple[str, str, str]] = set()
    deduped: list[Example] = []
    for example in examples:
        key = (example.parent_url, example.target_url, example.anchor)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(example)
    examples = deduped

    counts = label_counts([example.label for example in examples])
    if len(examples) < 200:
        raise SystemExit(f"Need at least 200 labeled examples, got {len(examples)}")
    if counts.get("positive", 0) < 50:
        raise SystemExit(f"Need at least 50 positive examples, got {counts.get('positive', 0)}")

    labels = [example.label for example in examples]
    groups = [example.target_host or f"id:{example.id}" for example in examples]
    train_idx, valid_idx, split_method = group_split(
        labels, groups, test_size=args.test_size, random_state=args.random_state
    )
    train = [examples[i] for i in train_idx]
    valid = [examples[i] for i in valid_idx]

    model = build_model()
    model.fit([example.text for example in train], [example.label for example in train])
    metrics = evaluate(
        model,
        [example.text for example in valid],
        [example.label for example in valid],
        with_thresholds=True,
    )
    selected_threshold = select_threshold(metrics.get("thresholds", {}))

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / DEFAULT_MODEL_NAME
    metrics_path = args.out / "link_metrics.json"
    predictions_path = args.out / "link_validation_predictions.jsonl"

    final_model = build_model()
    final_model.fit(
        [example.text for example in examples],
        [example.label for example in examples],
    )

    save_bundle(
        model_path,
        {
            "model": final_model,
            "feature_fields": [
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
            ],
            "feature_policy": "linkverdict_text_metadata_without_soft_link_heuristics",
            "labels": ["negative", "positive"],
            "ignored_rating": 3,
            "training_examples": len(examples),
            "positive_threshold": selected_threshold,
        },
    )
    write_validation_predictions(model, valid, predictions_path, selected_threshold)

    report = {
        "db": str(args.db),
        "crawl_dbs": [str(path) for path in args.crawl_db or []],
        "hand_examples": hand_count,
        "outcome_examples": outcome_count,
        "examples": len(examples),
        "train_examples": len(train),
        "validation_examples": len(valid),
        "split_method": split_method,
        "label_counts": counts,
        "train_label_counts": label_counts([example.label for example in train]),
        "validation_label_counts": label_counts([example.label for example in valid]),
        "unique_target_hosts": len({example.target_host for example in examples}),
        "positive_target_hosts": host_label_counts(examples, "positive"),
        "metrics": metrics,
        "selected_threshold": selected_threshold,
        "selected_threshold_metrics": metrics["thresholds"][f"{selected_threshold:.2f}"],
        "final_model_training_examples": len(examples),
        "model_path": str(model_path),
        "validation_predictions_path": str(predictions_path),
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
