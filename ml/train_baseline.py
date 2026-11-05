"""
ScamShield baseline: Logistic Regression on Enron Spam Dataset.

Usage (from the ml/ folder):
    python train_baseline.py
    python train_baseline.py --test

Outputs:
    data/splits.csv
    results/baseline_metrics.csv
    results/cm_<feature_set>_<split>.png
    models/baseline_app.joblib
"""

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "enron_spam_data_new.csv"
SPLITS_PATH = ROOT / "data" / "splits.csv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

SEED = 42

NUMERIC = [
    "Subject Length",
    "Subject Word Count",
    "Subject Uppercase Count",
    "Subject Lowercase Count",
    "Subject Digit Count",
    "Subject Special Character Count",
    "Subject Exclamation Count",
    "Subject Question Mark Count",
    "Subject Dollar Sign Count",
    "Subject Average Word Length",
    "Subject Uppercase Ratio",

    "Message Length",
    "Message Word Count",
    "Message Uppercase Count",
    "Message Lowercase Count",
    "Message Digit Count",
    "Message Special Character Count",
    "Message Exclamation Count",
    "Message Question Mark Count",
    "Message Dollar Sign Count",
    "Message Average Word Length",
    "Message Uppercase Ratio",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["Spam/Ham"]).copy()

    df["Subject"] = df["Subject"].fillna("").astype(str)
    df["Message"] = df["Message"].fillna("").astype(str)

    df["label"] = df["Spam/Ham"].astype(int)

    df[NUMERIC] = df[NUMERIC].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)

    df["text"] = df["Subject"] + " " + df["Message"]

    df = df.reset_index(drop=True)
    df["row_id"] = df.index

    return df


def make_splits(df: pd.DataFrame):
    """Stratified 70 / 15 / 15 split with a fixed seed."""

    train, temp = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=SEED
    )

    val, test = train_test_split(
        temp,
        test_size=0.50,
        stratify=temp["label"],
        random_state=SEED
    )

    splits = pd.concat([
        train[["row_id"]].assign(split="train"),
        val[["row_id"]].assign(split="val"),
        test[["row_id"]].assign(split="test"),
    ])

    splits.to_csv(SPLITS_PATH, index=False)

    return train, val, test


def text_vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
        sublinear_tf=True
    )


def feature_sets():
    """
    text_only:    Subject + Message using TF-IDF
    subject_only: Subject using TF-IDF
    numeric_only: All 22 engineered numerical features
    combined:     TF-IDF + all 22 numerical features
    """

    return {
        "text_only": ColumnTransformer([
            ("text", text_vectorizer(), "text")
        ]),

        "subject_only": ColumnTransformer([
            ("text", text_vectorizer(), "Subject")
        ]),

        "numeric_only": ColumnTransformer([
            ("num", StandardScaler(), NUMERIC)
        ]),

        "combined": ColumnTransformer([
            ("text", text_vectorizer(), "text"),
            ("num", StandardScaler(), NUMERIC)
        ]),
    }


def evaluate(name, model, X, y, split_name):

    pred = model.predict(X)

    cm = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()

    disp = ConfusionMatrixDisplay(
        cm,
        display_labels=["ham", "spam"]
    )

    disp.plot(cmap="Blues", colorbar=False)

    plt.title(f"Logistic regression: {name} ({split_name})")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / f"cm_{name}_{split_name}.png",
        dpi=120
    )

    plt.close()

    return {
        "feature_set": name,
        "split": split_name,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "false_negatives": int(fn),
        "false_positives": int(fp),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        action="store_true",
        help="also evaluate on the held-out test set"
    )

    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_data()

    train, val, test = make_splits(df)

    print(
        f"Rows: {len(df)} "
        f"train={len(train)} "
        f"val={len(val)} "
        f"test={len(test)}"
    )

    print(f"Spam share: {df['label'].mean():.1%}\n")

    rows = []

    for name, features in feature_sets().items():

        model = Pipeline([
            ("features", features),
            ("clf", LogisticRegression(
                max_iter=2000,
                random_state=SEED
            ))
        ])

        model.fit(train, train["label"])

        rows.append(
            evaluate(name, model, val, val["label"], "val")
        )

        if args.test:
            rows.append(
                evaluate(name, model, test, test["label"], "test")
            )

        if name == "text_only":
            joblib.dump(
                model,
                MODELS_DIR / "baseline_app.joblib"
            )

    results = pd.DataFrame(rows)

    results.to_csv(
        RESULTS_DIR / "baseline_metrics.csv",
        index=False
    )

    with pd.option_context(
        "display.float_format", "{:.3f}".format,
        "display.width", 120
    ):
        print(results.to_string(index=False))

    print(f"\nSaved metrics and confusion matrices to {RESULTS_DIR}")
    print(f"Saved app model to {MODELS_DIR / 'baseline_app.joblib'}")


if __name__ == "__main__":
    main()
