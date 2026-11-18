"""Model 1 – Klasik baseline (Alan B): TF-IDF + Lojistik Regresyon.
Çalıştır: python -m src.train_baseline
"""
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.common import load_labels, load_split  # noqa: E402


def main():
    train, val = load_split("train"), load_split("val")
    assert train["label_id"].nunique() == len(load_labels()), "Her sınıf eğitimde bulunmalı"

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("clf", LogisticRegression(C=10, max_iter=3000)),
    ])
    pipe.fit(train["clean"], train["label_id"])
    acc = (pipe.predict(val["clean"]) == val["label_id"]).mean()
    print(f"TF-IDF + LogReg  val accuracy (eşiksiz): {acc:.4f}")
    joblib.dump(pipe, config.ARTIFACTS / "tfidf_logreg.joblib")


if __name__ == "__main__":
    main()
