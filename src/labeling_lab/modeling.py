from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def build_tfidf_logreg(
    *,
    word_ngram_range: tuple[int, int],
    char_ngram_range: tuple[int, int],
    max_iter: int,
) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(ngram_range=word_ngram_range, min_df=2, max_df=0.95, sublinear_tf=True, strip_accents="unicode")),
                        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngram_range, min_df=2, max_df=0.98, sublinear_tf=True, strip_accents="unicode")),
                    ]
                ),
            ),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=max_iter, random_state=13)),
        ]
    )


def save_bundle(path: Path, bundle: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
