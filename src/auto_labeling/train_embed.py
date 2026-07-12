"""Embedding-Variante: eingefrorener Encoder + LogReg, Vergleich zur TF-IDF-Baseline.

Kurs-Regel (INFO4271 FAQ 5): kein Retrieval-trainiertes Modell — deshalb
paraphrase-multilingual-MiniLM-L12-v2 (Paraphrase-Training) statt BGE-M3/E5.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .train import load_examples

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CACHE = Path("data/cache/embeddings")


def embed(texts: list[str], tag: str) -> np.ndarray:
    cache_file = CACHE / f"{tag}-{len(texts)}.npy"
    if cache_file.exists():
        return np.load(cache_file)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    vecs = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, vecs)
    return vecs


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

    train = load_examples(Path("data/labels.final.jsonl"))
    hold = load_examples(Path("data/eval_holdout.jsonl"))
    x_train = embed([str(e["text"]) for e in train], "train")
    x_hold = embed([str(e["text"]) for e in hold], "holdout")
    y_train = [str(e["label"]) for e in train]
    y_hold = [str(e["label"]) for e in hold]

    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=13)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_hold)
    print(f"holdout acc: {accuracy_score(y_hold, pred):.3f}")
    print(classification_report(y_hold, pred))

    import joblib

    out = Path("data/models/page_verdict_embed.joblib")
    joblib.dump({"model": clf, "encoder": MODEL_NAME}, out)
    metrics = {
        "encoder": MODEL_NAME,
        "examples": len(train),
        "holdout_examples": len(hold),
        "holdout_accuracy": float(accuracy_score(y_hold, pred)),
    }
    Path("data/models/embed_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
