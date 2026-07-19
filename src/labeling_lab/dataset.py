from __future__ import annotations

import hashlib
import json
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from sklearn.model_selection import StratifiedGroupKFold

from .link_features import LinkVerdictInput, host_from_url, make_text as make_link_text

LABELS = {"negative", "positive"}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def normalize_url(value: object) -> str:
    url = str(value or "").strip()
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower().removeprefix("www.")
        if not host:
            return url
        port = f":{parsed.port}" if parsed.port else ""
        path = parsed.path.rstrip("/") or "/"
        return urlunsplit(((parsed.scheme or "https").lower(), host + port, path, parsed.query, ""))
    except ValueError:
        return url


def page_text(fields: dict[str, object]) -> str:
    return "\n".join(
        (
            f"title: {_space(fields.get('title'))}",
            f"url: {_space(fields.get('url'))}",
            f"display_url: {_space(fields.get('display_url'))}",
            f"snippet: {_space(fields.get('snippet'))}",
            f"text: {_space(fields.get('text'))[:3000]}",
        )
    )


def _space(value: object) -> str:
    return " ".join(str(value or "").split())


def _page_snippet(snapshot: dict[str, object]) -> str:
    description = _space(snapshot.get("description"))
    h1 = _space(snapshot.get("h1"))
    parts = (description, h1 if h1 not in description else "")
    return " ".join(part for part in parts if part)[:700]


def _display_url(url: str) -> str:
    return url.replace("https://", "").replace("http://", "").strip("/")


def _valid_teacher_label(row: dict[str, object]) -> bool:
    if not row.get("teacher"):
        return True
    if float(row.get("confidence") or 0.0) < 0.7:
        return False
    if row.get("label") != "positive":
        return True
    return (
        row.get("english") is True
        and row.get("tuebingen_related") is True
        and row.get("general_search_useful") is True
        and row.get("research_only") is False
    )


def _source_paths(root: Path, section: dict[str, object], key: str) -> list[Path]:
    values = section.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{key} must be a non-empty list")
    paths = [root / str(value) for value in values]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing training inputs: " + ", ".join(missing))
    return paths


def _dedupe(
    rows: list[dict[str, object]], *, identity_key: str, content_key: str
) -> tuple[list[dict[str, object]], dict[str, int]]:
    excluded = Counter()
    unique = rows
    for key in (identity_key, content_key):
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in unique:
            grouped[str(row[key])].append(row)
        unique = []
        for _, copies in sorted(grouped.items()):
            labels = {str(row["label"]) for row in copies}
            if len(labels) != 1:
                excluded[f"conflicting_{key}"] += len(copies)
                continue
            unique.append(
                max(copies, key=lambda row: (int(row["completeness"]), str(row["source"])))
            )
            excluded[f"duplicate_{key}"] += len(copies) - 1
    for row in unique:
        row.pop("completeness", None)
        row.pop("content_key", None)
    return unique, dict(excluded)


def build_page_dataset(
    root: Path, section: dict[str, object]
) -> tuple[list[dict[str, object]], dict[str, int]]:
    prompt_version = str(section.get("prompt_version") or "")
    label_paths = _source_paths(root, section, "label_files")
    snapshot_paths = _source_paths(root, section, "snapshot_files")
    snapshots_by_hash: dict[str, dict[str, object]] = {}
    snapshots_by_url: dict[str, dict[str, object]] = {}
    for path in snapshot_paths:
        for row in read_jsonl(path):
            if not row.get("text"):
                continue
            text_hash = str(row.get("text_hash") or "")
            url = normalize_url(row.get("normalized_url") or row.get("url"))
            if text_hash:
                snapshots_by_hash.setdefault(text_hash, row)
            if url:
                snapshots_by_url.setdefault(url, row)

    rows: list[dict[str, object]] = []
    excluded = Counter()
    for path in label_paths:
        for label in read_jsonl(path):
            if label.get("prompt_version") != prompt_version:
                excluded["wrong_prompt_version"] += 1
                continue
            if label.get("label") not in LABELS:
                excluded["non_binary_label"] += 1
                continue
            if not _valid_teacher_label(label):
                excluded["inconsistent_teacher_label"] += 1
                continue
            url = normalize_url(label.get("normalized_url") or label.get("url"))
            text_hash = str(label.get("text_hash") or "")
            snapshot = snapshots_by_hash.get(text_hash) or snapshots_by_url.get(url)
            if not url or snapshot is None:
                excluded["missing_snapshot"] += 1
                continue
            body = _space(snapshot.get("text"))
            if not body:
                excluded["empty_text"] += 1
                continue
            content_hash = text_hash or stable_hash(body.lower())
            fields = {
                "title": _space(
                    snapshot.get("title") or snapshot.get("serp_title") or label.get("title")
                ),
                "url": str(snapshot.get("url") or label.get("url") or url),
                "display_url": _display_url(str(snapshot.get("url") or label.get("url") or url)),
                "snippet": _page_snippet(snapshot),
                "text": body[:3000],
            }
            group = host_from_url(url)
            if not group:
                excluded["missing_host"] += 1
                continue
            rows.append(
                {
                    "id": stable_hash(["page", url, content_hash])[:20],
                    "task": "page",
                    "label": label["label"],
                    "group": group,
                    "url": url,
                    "source": str(label.get("source_file") or path.relative_to(root)),
                    "prompt_version": prompt_version,
                    "prompt_sha256": str(label.get("prompt_sha256") or ""),
                    "teacher": str(label.get("teacher") or ""),
                    "teacher_model": str(label.get("model") or ""),
                    "confidence": float(label.get("confidence") or 0.0),
                    **({"training_only": True} if label.get("training_only") else {}),
                    "text": page_text(fields),
                    "identity": url,
                    "content_key": content_hash,
                    "completeness": sum(bool(fields[key]) for key in fields),
                }
            )
    deduped, dedupe_counts = _dedupe(rows, identity_key="identity", content_key="content_key")
    excluded.update(dedupe_counts)
    return deduped, dict(excluded)


def build_link_dataset(
    root: Path, section: dict[str, object]
) -> tuple[list[dict[str, object]], dict[str, int]]:
    prompt_version = str(section.get("prompt_version") or "")
    force_negative_hosts = {
        str(host).lower().removeprefix("www.")
        for host in section.get("force_negative_hosts", [])
    }
    label_paths = (
        _source_paths(root, section, "label_files")
        if section.get("use_teacher_labels", True)
        else []
    )
    review_paths = (
        _source_paths(root, section, "human_review_files")
        if section.get("human_review_files")
        else []
    )
    review_sets = {str(value) for value in section.get("human_review_sets", [])}
    training_sets = {str(value) for value in section.get("human_training_sets", [])}
    reviews = [
        (path, review)
        for path in review_paths
        for review in read_jsonl(path)
    ]
    human_labels: dict[str, str] = {}
    for _, review in reviews:
        if review_sets and str(review.get("review_set") or "") not in review_sets:
            continue
        content_hash = str(review.get("target_text_hash") or "")
        label = str(review.get("label") or "")
        if not content_hash or label not in LABELS:
            continue
        if content_hash in human_labels and human_labels[content_hash] != label:
            raise ValueError(f"Conflicting human reviews for {content_hash}")
        human_labels[content_hash] = label
    label_rows = [
        (path, label)
        for path in label_paths
        for label in read_jsonl(path)
    ]
    label_rows.extend(
        (
            path,
            {
                **review,
                "prompt_version": prompt_version,
                "confidence": 1.0,
                "teacher": "",
                "_human_review": True,
            },
        )
        for path, review in reviews
        if str(review.get("review_set") or "") in training_sets
        and str(review.get("label") or "") in LABELS
    )
    rows: list[dict[str, object]] = []
    excluded = Counter()
    for path, label in label_rows:
        if label.get("prompt_version") != prompt_version:
            excluded["wrong_prompt_version"] += 1
            continue
        if label.get("label") not in LABELS:
            excluded["non_binary_label"] += 1
            continue
        if not _valid_teacher_label(label):
            excluded["inconsistent_teacher_label"] += 1
            continue
        raw_target_url = str(label.get("target_url") or label.get("url") or "")
        raw_parent_url = str(label.get("parent_url") or "")
        target_url = normalize_url(raw_target_url)
        parent_url = normalize_url(raw_parent_url)
        anchor = _space(label.get("anchor"))
        group = host_from_url(target_url)
        if not target_url or not group:
            excluded["missing_target"] += 1
            continue
        identity = stable_hash([parent_url, target_url, anchor])
        original_label = str(label["label"])
        content_hash = str(label.get("target_text_hash") or identity)
        reviewed_label = human_labels.get(content_hash, original_label)
        human_reviewed = bool(label.get("_human_review")) or content_hash in human_labels
        final_label = (
            reviewed_label
            if human_reviewed
            else "negative"
            if group in force_negative_hosts
            else original_label
        )
        if final_label != original_label:
            excluded[
                "policy_relabels" if group in force_negative_hosts else "human_review_relabels"
            ] += 1
        model_input = LinkVerdictInput(
            anchor=anchor,
            target_url=raw_target_url,
            parent_url=raw_parent_url,
            parent_host=str(label.get("parent_host") or ""),
            parent_depth=_optional_int(label.get("parent_depth")),
            parent_pageverdict_score=_optional_float(label.get("parent_pageverdict_score")),
            parent_pageverdict_decision=str(label.get("parent_pageverdict_decision") or ""),
            parent_relevance=_optional_float(label.get("parent_relevance")),
            target_host=str(label.get("target_host") or group),
            target_depth=_optional_int(label.get("target_depth")),
            raw_score=_optional_float(label.get("raw_score")),
        )
        rows.append(
            {
                "id": stable_hash(["link", identity, content_hash])[:20],
                "task": "link",
                "label": final_label,
                "original_label": original_label,
                "label_correction": (
                    "force_negative_host"
                    if group in force_negative_hosts and final_label != original_label
                    else "human_review"
                    if reviewed_label != original_label
                    else ""
                ),
                "group": group,
                "url": target_url,
                "source": str(path.relative_to(root)),
                "prompt_version": prompt_version,
                "prompt_sha256": str(label.get("prompt_sha256") or ""),
                "teacher": str(label.get("teacher") or ""),
                "teacher_model": str(label.get("model") or ""),
                "label_source": "human" if human_reviewed else "teacher",
                "confidence": float(label.get("confidence") or 0.0),
                "stratum": str(label.get("selection") or label.get("sample_bucket") or ""),
                "text": make_link_text(model_input),
                "identity": identity,
                "content_key": content_hash,
                "completeness": sum(
                    value not in {None, ""}
                    for value in (
                        anchor,
                        parent_url,
                        label.get("parent_pageverdict_score"),
                        label.get("parent_relevance"),
                    )
                ),
            }
        )
    deduped, dedupe_counts = _dedupe(rows, identity_key="identity", content_key="content_key")
    excluded.update(dedupe_counts)
    return deduped, dict(excluded)


def _optional_int(value: object) -> int | None:
    return None if value in {None, ""} else int(value)


def _optional_float(value: object) -> float | None:
    return None if value in {None, ""} else float(value)


def split_dataset(rows: list[dict[str, object]], seed: int) -> list[dict[str, object]]:
    base = [row for row in rows if not row.get("training_only")]
    additions = [row for row in rows if row.get("training_only")]
    labels = [str(row["label"]) for row in base]
    groups = [str(row["group"]) for row in base]
    if len(base) < 100 or len(set(labels)) != 2 or len(set(groups)) < 10:
        raise ValueError("Dataset needs at least 100 examples, both labels, and 10 groups")
    indices = list(range(len(base)))
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    remaining, test = _most_balanced_fold(outer, indices, labels, groups)
    inner_labels = [labels[index] for index in remaining]
    inner_groups = [groups[index] for index in remaining]
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    train_local, validation_local = _most_balanced_fold(
        inner, list(remaining), inner_labels, inner_groups
    )
    split_by_index = {int(index): "test" for index in test}
    split_by_index.update({int(remaining[index]): "train" for index in train_local})
    split_by_index.update({int(remaining[index]): "validation" for index in validation_local})
    result = [{**row, "split": split_by_index[index]} for index, row in enumerate(base)]
    held_out_groups = {
        str(row["group"])
        for row in result
        if row["split"] in {"validation", "test"}
    }
    result.extend(
        {**row, "split": "train"}
        for row in additions
        if str(row["group"]) not in held_out_groups
    )
    split_groups = {
        split: {str(row["group"]) for row in result if row["split"] == split}
        for split in ("train", "validation", "test")
    }
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    if any(split_groups[left] & split_groups[right] for left, right in pairs):
        raise AssertionError("Host leakage between dataset splits")
    return sorted(result, key=lambda row: (str(row["split"]), str(row["id"])))


def _most_balanced_fold(splitter, rows, labels, groups):
    label_counts = Counter(labels)
    target_size = len(rows) / splitter.n_splits

    def imbalance(candidate) -> float:
        _, held_out = candidate
        held_out_counts = Counter(labels[index] for index in held_out)
        score = abs(len(held_out) - target_size) / target_size
        for label, count in label_counts.items():
            target = count / splitter.n_splits
            score += abs(held_out_counts[label] - target) / target
        return score

    return min(splitter.split(rows, labels, groups), key=imbalance)


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "examples": len(rows),
        "groups": len({row["group"] for row in rows}),
        "labels": dict(Counter(str(row["label"]) for row in rows)),
        "splits": {
            split: {
                "examples": sum(row["split"] == split for row in rows),
                "groups": len({row["group"] for row in rows if row["split"] == split}),
                "labels": dict(Counter(str(row["label"]) for row in rows if row["split"] == split)),
            }
            for split in ("train", "validation", "test")
        },
    }


def prepare(config_path: Path) -> Path:
    root = config_path.resolve().parent
    with config_path.open("rb") as file:
        config = tomllib.load(file)
    if config.get("schema_version") != 1:
        raise ValueError("training.toml schema_version must be 1")
    seed = int(config.get("random_seed", 0))
    output = root / str(config.get("output_dir") or "data/release")
    inputs: list[Path] = []
    manifest: dict[str, object] = {
        "schema_version": 1,
        "random_seed": seed,
        "config_sha256": file_hash(config_path),
        "tasks": {},
    }
    for task, builder in (("page", build_page_dataset), ("link", build_link_dataset)):
        section = config.get(task)
        if not isinstance(section, dict):
            raise ValueError(f"Missing [{task}] section")
        if task != "link" or section.get("use_teacher_labels", True):
            inputs.extend(_source_paths(root, section, "label_files"))
        if task == "link" and section.get("human_review_files"):
            inputs.extend(_source_paths(root, section, "human_review_files"))
        if task == "page":
            inputs.extend(_source_paths(root, section, "snapshot_files"))
        threshold_file = section.get("threshold_development_file")
        if threshold_file:
            threshold_path = root / str(threshold_file)
            if not threshold_path.is_file():
                raise FileNotFoundError(f"Missing threshold development input: {threshold_path}")
            inputs.append(threshold_path)
        rows, excluded = builder(root, section)
        rows = split_dataset(rows, seed)
        split_hashes = {}
        for split in ("train", "validation", "test"):
            path = output / "datasets" / task / f"{split}.jsonl"
            write_jsonl(path, [row for row in rows if row["split"] == split])
            split_hashes[split] = file_hash(path)
        task_manifest = {
            **_summary(rows),
            "excluded": excluded,
            "split_sha256": split_hashes,
            "dataset_sha256": stable_hash(split_hashes),
        }
        if "positive_threshold" in section:
            threshold = float(section["positive_threshold"])
            if not 0 < threshold < 1 or not threshold_file:
                raise ValueError(
                    f"{task} positive_threshold needs a value between 0 and 1 and "
                    "threshold_development_file"
                )
            task_manifest["positive_threshold"] = threshold
            task_manifest["threshold_development_sha256"] = file_hash(threshold_path)
            task_manifest["threshold_selection"] = str(
                section.get("threshold_selection") or "configured"
            )
        manifest["tasks"][task] = task_manifest
    manifest["inputs"] = {
        str(path.relative_to(root)): file_hash(path) for path in sorted(set(inputs))
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output
