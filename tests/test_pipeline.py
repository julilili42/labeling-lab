import json
from pathlib import Path

from labeling_lab.dataset import build_link_dataset, prepare, read_jsonl, split_dataset
from labeling_lab.release_benchmark import benchmark_release
from labeling_lab.training import train_release


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_release_pipeline_keeps_hosts_out_of_multiple_splits(tmp_path):
    page_labels = []
    snapshots = []
    link_labels = []
    for host_index in range(60):
        label = "positive" if host_index % 3 == 0 else "negative"
        for page_index in range(2):
            url = f"https://host-{host_index}.test/page-{page_index}"
            text_hash = f"page-{host_index}-{page_index}"
            page_labels.append(
                {
                    "url": url,
                    "normalized_url": url,
                    "text_hash": text_hash,
                    "prompt_version": "page-test",
                    "label": label,
                }
            )
            snapshots.append(
                {
                    "url": url,
                    "normalized_url": url,
                    "text_hash": text_hash,
                    "title": f"Page {host_index} {page_index}",
                    "description": "Tuebingen guide" if label == "positive" else "Unrelated page",
                    "text": (
                        "Useful Tuebingen visitor information "
                        if label == "positive"
                        else "Unrelated content "
                    )
                    + text_hash,
                }
            )
            link_labels.append(
                {
                    "parent_url": f"https://parent-{host_index}.test/",
                    "target_url": url,
                    "target_host": f"host-{host_index}.test",
                    "anchor": "Tuebingen guide" if label == "positive" else "Other",
                    "target_text_hash": text_hash,
                    "prompt_version": "link-test",
                    "label": label,
                }
            )
    _write(tmp_path / "page-labels.jsonl", page_labels)
    _write(tmp_path / "snapshots.jsonl", snapshots)
    _write(tmp_path / "link-labels.jsonl", link_labels)
    _write(
        tmp_path / "human-reviews.jsonl",
        [
            {
                "target_text_hash": "page-3-0",
                "review_set": "link-boundaries",
                "label": "negative",
            },
            {
                "target_text_hash": "page-2-0",
                "review_set": "unrelated-holdout",
                "label": "positive",
            },
            {
                "target_text_hash": "human-only",
                "review_set": "link-targeted",
                "label": "positive",
                "target_url": "https://human-only.test/guide",
                "parent_url": "https://parent.test/",
                "anchor": "Tuebingen guide",
            },
        ],
    )
    (tmp_path / "threshold-report.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "training.toml").write_text(
        """
schema_version = 1
random_seed = 13
output_dir = "release"
[page]
prompt_version = "page-test"
label_files = ["page-labels.jsonl"]
snapshot_files = ["snapshots.jsonl"]
[link]
prompt_version = "link-test"
label_files = ["link-labels.jsonl"]
human_review_files = ["human-reviews.jsonl"]
human_review_sets = ["link-boundaries"]
human_training_sets = ["link-targeted"]
positive_threshold = 0.75
threshold_development_file = "threshold-report.json"
force_negative_hosts = ["host-0.test"]
""".strip(),
        encoding="utf-8",
    )

    release = prepare(tmp_path / "training.toml")
    for task in ("page", "link"):
        rows = [
            row
            for split in ("train", "validation", "test")
            for row in read_jsonl(release / "datasets" / task / f"{split}.jsonl")
        ]
        groups = {
            split: {row["group"] for row in rows if row["split"] == split}
            for split in ("train", "validation", "test")
        }
        assert not groups["train"] & groups["validation"]
        assert not groups["train"] & groups["test"]
        assert not groups["validation"] & groups["test"]

    link_rows = [
        row
        for split in ("train", "validation", "test")
        for row in read_jsonl(release / "datasets" / "link" / f"{split}.jsonl")
    ]
    corrected = [row for row in link_rows if row["group"] == "host-0.test"]
    assert {row["label"] for row in corrected} == {"negative"}
    assert {row["original_label"] for row in corrected} == {"positive"}
    assert {row["label_correction"] for row in corrected} == {"force_negative_host"}
    reviewed = [row for row in link_rows if row["url"].endswith("host-3.test/page-0")]
    assert {row["label"] for row in reviewed} == {"negative"}
    assert {row["original_label"] for row in reviewed} == {"positive"}
    assert {row["label_correction"] for row in reviewed} == {"human_review"}
    human_only = [row for row in link_rows if row["group"] == "human-only.test"]
    assert len(human_only) == 1
    assert human_only[0]["label"] == "positive"
    assert human_only[0]["teacher"] == ""

    training = train_release(release)
    assert training["link"]["positive_threshold"] == 0.75
    assert training["link"]["threshold_source"] == "configured_development_set"
    report = benchmark_release(release)

    assert set(report) == {"page", "link"}
    assert (release / "benchmark" / "metrics.csv").is_file()
    assert (release / "benchmark" / "predictions.csv").is_file()


def test_link_dataset_can_use_only_human_reviews(tmp_path):
    _write(
        tmp_path / "reviews.jsonl",
        [
            {
                "review_set": "human-training",
                "label": "positive",
                "target_text_hash": "content-1",
                "target_url": "https://human.test/guide",
                "parent_url": "https://parent.test/",
                "anchor": "Tuebingen guide",
            }
        ],
    )
    rows, _ = build_link_dataset(
        tmp_path,
        {
            "prompt_version": "link-test",
            "use_teacher_labels": False,
            "human_review_files": ["reviews.jsonl"],
            "human_training_sets": ["human-training"],
        },
    )

    assert len(rows) == 1
    assert rows[0]["label"] == "positive"
    assert rows[0]["label_source"] == "human"


def test_training_only_rows_cannot_change_validation_or_test():
    base = [
        {
            "id": f"base-{host}-{page}",
            "label": "positive" if host % 3 == 0 else "negative",
            "group": f"host-{host}.test",
        }
        for host in range(60)
        for page in range(2)
    ]
    additions = [
        {
            "id": f"hard-{host}",
            "label": "negative",
            "group": f"host-{host}.test",
            "training_only": True,
        }
        for host in range(20)
    ]

    original = split_dataset(base, 13)
    augmented = split_dataset(base + additions, 13)

    for split in ("validation", "test"):
        assert {row["id"] for row in augmented if row["split"] == split} == {
            row["id"] for row in original if row["split"] == split
        }
    held_out_groups = {
        row["group"] for row in original if row["split"] in {"validation", "test"}
    }
    assert {row["id"] for row in augmented if row["id"].startswith("hard-")} == {
        row["id"] for row in additions if row["group"] not in held_out_groups
    }
