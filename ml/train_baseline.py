"""
ScamShield baseline: logistic regression on three feature sets.

Usage (from the ml/ folder):
    python train_baseline.py          # train + report on the validation set
    python train_baseline.py --test   # also report on the held-out test set (use for the final report only)

Outputs:
    data/splits.csv                 email_id -> train / val / test (everyone uses the same split)
    results/baseline_metrics.csv    accuracy, precision, recall, F1 per feature set
    results/cm_<feature_set>.png    confusion matrices
    models/baseline_app.joblib      text model used by the web app
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "spam_email_dataset.csv"
SPLITS_PATH = ROOT / "data" / "splits.csv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
SEED = 42

NUMERIC = [
    "num_words",
    "num_characters",
    "num_links",
    "has_suspicious_link",
    "num_attachments",
    "has_attachment",
    "sender_reputation_score",
    "email_hour",
    "is_weekend",
    "num_recipients",
    "contains_money_terms",
    "contains_urgency_terms",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # num_exclamation_marks is 0 for every row, so it carries no information
    df = df.drop(columns=["num_exclamation_marks"])
    df = df.drop_duplicates(subset=["subject", "email_text", "sender_email"])
    df["text"] = df["subject"] + " " + df["email_text"]
    return df


def make_splits(df: pd.DataFrame):
    """Stratified 70 / 15 / 15 split with a fixed seed."""
    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=SEED)
    splits = pd.concat(
        [
            train[["email_id"]].assign(split="train"),
            val[["email_id"]].assign(split="val"),
            test[["email_id"]].assign(split="test"),
        ]
    )
    splits.to_csv(SPLITS_PATH, index=False)
    return train, val, test


def text_vectorizer():
    return TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def feature_sets():
    """
    A_all:            every usable column (includes columns that leak the label)
    B_no_leaky_meta:  drops subject, sender_email, sender_domain, but keeps body text
    C_metadata_only:  numeric metadata only, no text, no sender info (hardest / most realistic)
    app_text:         subject + body text only, which is what the web form collects
    """
    cat = OneHotEncoder(handle_unknown="ignore")
    return {
        "A_all": ColumnTransformer(
            [
                ("text", text_vectorizer(), "text"),
                ("num", StandardScaler(), NUMERIC),
                ("cat", cat, ["sender_domain", "email_day_of_week"]),
            ]
        ),
        "B_no_leaky_meta": ColumnTransformer(
            [
                ("text", text_vectorizer(), "email_text"),
                ("num", StandardScaler(), NUMERIC),
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["email_day_of_week"]),
            ]
        ),
        "C_metadata_only": ColumnTransformer(
            [
                ("num", StandardScaler(), NUMERIC),
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["email_day_of_week"]),
            ]
        ),
        "app_text": ColumnTransformer([("text", text_vectorizer(), "text")]),
    }


def evaluate(name, model, X, y, split_name):
    pred = model.predict(X)
    cm = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()
    disp = ConfusionMatrixDisplay(cm, display_labels=["legit", "phishing"])
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"Logistic regression: {name} ({split_name})")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"cm_{name}_{split_name}.png", dpi=120)
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
    parser.add_argument("--test", action="store_true", help="also evaluate on the held-out test set")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_data()
    train, val, test = make_splits(df)
    print(f"Rows: {len(df)}  train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"Phishing share: {df['label'].mean():.1%}\n")

    rows = []
    for name, features in feature_sets().items():
        model = Pipeline(
            [("features", features), ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]
        )
        model.fit(train, train["label"])
        rows.append(evaluate(name, model, val, val["label"], "val"))
        if args.test:
            rows.append(evaluate(name, model, test, test["label"], "test"))
        if name == "app_text":
            joblib.dump(model, MODELS_DIR / "baseline_app.joblib")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 120):
        print(results.to_string(index=False))
    print(f"\nSaved metrics and confusion matrices to {RESULTS_DIR}")
    print(f"Saved app model to {MODELS_DIR / 'baseline_app.joblib'}")


if __name__ == "__main__":
    main()
