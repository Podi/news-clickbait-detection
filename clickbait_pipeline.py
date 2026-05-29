"""Reusable helpers for the clickbait binary-classification pilot.

The module keeps data loading, normalization, modeling, evaluation, feature
inspection, error analysis, and plotting in one place so the notebook can stay
short and presentation-focused.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
WEBIS_DIR = DATA_DIR / "clickbait17-train-170630"
KAGGLE_PATH = DATA_DIR / "kaggle" / "clickbait_data.csv"
RANDOM_STATE = 42
WEBIS_SCORE_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

NORMALIZED_COLUMNS = [
    "dataset",
    "id",
    "text_original",
    "text_clean",
    "label_raw",
    "score_raw",
    "label_binary",
]

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
APOSTROPHE_PATTERN = re.compile(r"[`´‘’]")
CONTRACTION_REPLACEMENTS = {
    "won't": "will not",
    "can't": "can not",
    "cannot": "can not",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'m": " am",
    "'d": " would",
    "'s": " is",
}


def read_jsonl(path: str | Path) -> pd.DataFrame:
    """Read a JSON Lines file into a DataFrame."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_webis_instances(path: str | Path = WEBIS_DIR / "instances.jsonl") -> pd.DataFrame:
    """Load Webis-Clickbait-17 instances."""
    return read_jsonl(path)


def load_webis_truth(path: str | Path = WEBIS_DIR / "truth.jsonl") -> pd.DataFrame:
    """Load Webis-Clickbait-17 truth labels and scores."""
    return read_jsonl(path)


def load_kaggle(path: str | Path = KAGGLE_PATH) -> pd.DataFrame:
    """Load the Kaggle clickbait CSV."""
    return pd.read_csv(path)


def inspect_webis_truth_fields(truth: pd.DataFrame) -> pd.DataFrame:
    """Summarize available Webis truth/score fields."""
    rows = []
    for column in truth.columns:
        series = truth[column]
        hashable_series = series.map(lambda value: json.dumps(value, sort_keys=True) if isinstance(value, list) else value)
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "unique_values": int(hashable_series.nunique(dropna=True)),
                "example": series.dropna().iloc[0] if series.notna().any() else None,
            }
        )
    return pd.DataFrame(rows)


def merge_webis_instances_truth(
    instances: pd.DataFrame,
    truth: pd.DataFrame,
    on: str = "id",
) -> pd.DataFrame:
    """Merge Webis post instances with their truth labels."""
    merged = instances.merge(truth, on=on, how="inner", validate="one_to_one")
    return merged


def clean_text(text: Any) -> str:
    """Lightly clean text before TF-IDF vectorization."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = URL_PATTERN.sub(" ", text)
    text = APOSTROPHE_PATTERN.sub("'", text)
    text = re.sub(r"\b(\w+)'ll\b", r"\1 will", text)
    text = re.sub(r"\b(\w+)\s+ll\b", r"\1 will", text)
    for contraction, replacement in CONTRACTION_REPLACEMENTS.items():
        text = text.replace(contraction, replacement)
    text = re.sub(r"\bll\b", "will", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def detect_kaggle_columns(kaggle: pd.DataFrame) -> tuple[str, str]:
    """Detect the headline/text column and binary label column in the Kaggle CSV."""
    lower_to_original = {column.lower().strip(): column for column in kaggle.columns}

    text_candidates = ("headline", "title", "text", "post", "content")
    label_candidates = ("clickbait", "label", "target", "class", "is_clickbait")

    text_column = next((lower_to_original[name] for name in text_candidates if name in lower_to_original), None)
    label_column = next((lower_to_original[name] for name in label_candidates if name in lower_to_original), None)

    if text_column is None:
        object_columns = kaggle.select_dtypes(include=["object", "string"]).columns.tolist()
        if object_columns:
            text_column = max(object_columns, key=lambda column: kaggle[column].astype(str).str.len().mean())

    if label_column is None:
        for column in kaggle.columns:
            values = set(kaggle[column].dropna().astype(str).str.strip().str.lower().unique())
            if values and values.issubset({"0", "1", "true", "false", "clickbait", "non_clickbait", "non-clickbait"}):
                label_column = column
                break

    if text_column is None or label_column is None:
        raise ValueError(
            "Could not detect Kaggle text and label columns. "
            f"Available columns: {kaggle.columns.tolist()}"
        )

    if text_column == label_column:
        raise ValueError(f"Detected the same Kaggle column for text and label: {text_column!r}")

    return text_column, label_column


def _join_text(value: Any) -> str:
    """Convert Webis list-like text fields to a single string."""
    if isinstance(value, list):
        return " ".join(str(item) for item in value if not pd.isna(item))
    if pd.isna(value):
        return ""
    return str(value)


def label_to_binary(label: Any) -> int:
    """Convert common raw labels to 1 for clickbait and 0 for non-clickbait."""
    if pd.isna(label):
        raise ValueError("Cannot convert missing label to binary.")
    if isinstance(label, (int, float)) and label in {0, 1}:
        return int(label)

    normalized = str(label).strip().lower()
    clickbait_values = {"1", "true", "clickbait", "yes"}
    non_clickbait_values = {"0", "false", "no-clickbait", "non_clickbait", "non-clickbait", "no"}

    if normalized in clickbait_values:
        return 1
    if normalized in non_clickbait_values:
        return 0
    raise ValueError(f"Unrecognized clickbait label: {label!r}")


def label_to_name(label: Any) -> str:
    """Convert a label to a consistent readable class name."""
    return "clickbait" if label_to_binary(label) == 1 else "non_clickbait"


def score_to_binary(score: Any, threshold: float = 0.5) -> int:
    """Convert a numeric clickbait score to a binary label."""
    if pd.isna(score):
        raise ValueError("Cannot convert missing score to binary.")
    return int(float(score) >= threshold)


def normalize_webis(
    merged_webis: pd.DataFrame,
    text_column: str = "postText",
    threshold: float | None = None,
) -> pd.DataFrame:
    """Normalize merged Webis records to the common project schema.

    By default, Webis uses the provided `truthClass`. Pass `threshold` to derive
    labels from `truthMean` instead, which is useful for threshold experiments.
    """
    frame = pd.DataFrame()
    frame["id"] = merged_webis["id"].astype(str)
    frame["dataset"] = "webis"
    frame["text_original"] = merged_webis.apply(
        lambda row: _join_text(row.get(text_column)) or _join_text(row.get("targetTitle")),
        axis=1,
    )
    frame["text_clean"] = frame["text_original"].apply(clean_text)
    frame["label_raw"] = merged_webis["truthClass"]
    frame["score_raw"] = pd.to_numeric(merged_webis["truthMean"], errors="coerce")
    if threshold is None:
        frame["label_binary"] = frame["label_raw"].apply(label_to_binary)
    else:
        frame["label_binary"] = frame["score_raw"].apply(lambda score: score_to_binary(score, threshold))
    return frame[NORMALIZED_COLUMNS]


def normalize_kaggle(
    kaggle: pd.DataFrame,
    text_column: str | None = None,
    label_column: str | None = None,
) -> pd.DataFrame:
    """Normalize Kaggle records to the common project schema."""
    if text_column is None or label_column is None:
        detected_text_column, detected_label_column = detect_kaggle_columns(kaggle)
        text_column = text_column or detected_text_column
        label_column = label_column or detected_label_column

    frame = pd.DataFrame()
    frame["id"] = [f"kaggle_{index}" for index in kaggle.index]
    frame["dataset"] = "kaggle"
    frame["text_original"] = kaggle[text_column].fillna("").astype(str)
    frame["text_clean"] = frame["text_original"].apply(clean_text)
    frame["label_raw"] = kaggle[label_column].apply(label_to_name)
    frame["score_raw"] = pd.to_numeric(kaggle[label_column], errors="coerce")
    frame["label_binary"] = frame["label_raw"].apply(label_to_binary)
    return frame[NORMALIZED_COLUMNS]


def load_normalized_webis(
    instances_path: str | Path = WEBIS_DIR / "instances.jsonl",
    truth_path: str | Path = WEBIS_DIR / "truth.jsonl",
    threshold: float | None = None,
) -> pd.DataFrame:
    """Load, merge, and normalize Webis in one call."""
    instances = load_webis_instances(instances_path)
    truth = load_webis_truth(truth_path)
    merged = merge_webis_instances_truth(instances, truth)
    return normalize_webis(merged, threshold=threshold)


def load_normalized_kaggle(path: str | Path = KAGGLE_PATH) -> pd.DataFrame:
    """Load and normalize Kaggle in one call."""
    return normalize_kaggle(load_kaggle(path))


def label_distribution(df: pd.DataFrame, label_column: str = "label_binary") -> pd.DataFrame:
    """Return label counts and percentages."""
    counts = df[label_column].value_counts(dropna=False).rename_axis(label_column).reset_index(name="count")
    counts["percentage"] = counts["count"] / len(df) * 100
    return counts


def example_rows_per_class(
    df: pd.DataFrame,
    n: int = 3,
    label_column: str = "label_binary",
) -> pd.DataFrame:
    """Return a few example rows per class for quick qualitative inspection."""
    columns = [column for column in NORMALIZED_COLUMNS if column in df.columns]
    return df.groupby(label_column, group_keys=False).head(n)[columns].reset_index(drop=True)


def eda_summary(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create compact EDA summaries for a normalized dataset."""
    summary = df.copy()
    summary["text_length_chars"] = summary["text_clean"].str.len()
    summary["text_length_words"] = summary["text_clean"].str.split().str.len()
    missing = df.isna().sum().rename("missing_count").reset_index()
    missing.columns = ["column", "missing_count"]

    return {
        "dataset_size": pd.DataFrame({"rows": [len(df)], "columns": [df.shape[1]]}),
        "missing": missing,
        "duplicate_text_count": pd.DataFrame(
            {"duplicate_text_count": [int(summary["text_clean"].duplicated().sum())]}
        ),
        "dataset_counts": summary["dataset"].value_counts(dropna=False).reset_index(name="count"),
        "label_counts": label_distribution(summary),
        "label_by_dataset": pd.crosstab(summary["dataset"], summary["label_binary"]),
        "text_lengths": summary.groupby("label_binary")[["text_length_chars", "text_length_words"]].describe(),
        "examples_per_class": example_rows_per_class(summary),
    }


def split_train_test(
    df: pd.DataFrame,
    text_column: str = "text_clean",
    label_column: str = "label_binary",
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create a stratified train/test split."""
    return train_test_split(
        df[text_column],
        df[label_column],
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_column],
    )


def make_tfidf_model(estimator: BaseEstimator) -> Pipeline:
    """Create a TF-IDF classification pipeline around an estimator."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 3),
                    max_features=30000,
                    min_df=2,
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            ("classifier", estimator),
        ]
    )


def baseline_models(random_state: int = RANDOM_STATE) -> dict[str, BaseEstimator]:
    """Return the baseline models used in the pilot."""
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_tfidf_model(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
        ),
        "linear_svm": make_tfidf_model(LinearSVC(class_weight="balanced", random_state=random_state)),
        "naive_bayes": make_tfidf_model(MultinomialNB()),
    }


def train_models(
    models: dict[str, BaseEstimator],
    x_train: Iterable[str],
    y_train: Iterable[int],
) -> dict[str, BaseEstimator]:
    """Fit all provided models and return them."""
    fitted: dict[str, BaseEstimator] = {}
    for name, model in models.items():
        fitted[name] = model.fit(x_train, y_train)
    return fitted


def _positive_scores(model: BaseEstimator, x: Iterable[str]) -> list[float] | None:
    """Return score-like values for ROC AUC when the model exposes them."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return None


def evaluate_model(model: BaseEstimator, x_test: Iterable[str], y_test: Iterable[int]) -> dict[str, Any]:
    """Evaluate one fitted model with common binary-classification metrics."""
    predictions = model.predict(x_test)
    scores = _positive_scores(model, x_test)

    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["non_clickbait", "clickbait"],
            zero_division=0,
        ),
        "classification_report_dict": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["non_clickbait", "clickbait"],
            zero_division=0,
            output_dict=True,
        ),
        "predictions": predictions,
    }

    if scores is not None and len(set(y_test)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_test, scores)
        metrics["scores"] = scores

    return metrics


def evaluate_models(
    models: dict[str, BaseEstimator],
    x_test: Iterable[str],
    y_test: Iterable[int],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Evaluate multiple fitted models and return a metric table plus details."""
    details = {name: evaluate_model(model, x_test, y_test) for name, model in models.items()}
    rows = []
    for name, result in details.items():
        rows.append(
            {
                "model": name,
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "macro_f1": result["macro_f1"],
                "roc_auc": result.get("roc_auc"),
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True), details


def select_best_model(
    models: dict[str, BaseEstimator],
    metrics_df: pd.DataFrame,
    metric: str = "macro_f1",
) -> tuple[str, BaseEstimator, pd.Series]:
    """Select the best fitted model from an evaluation table."""
    if metric not in metrics_df.columns:
        raise ValueError(f"Metric {metric!r} is not available. Columns: {metrics_df.columns.tolist()}")
    best_row = metrics_df.sort_values(metric, ascending=False).iloc[0]
    best_name = str(best_row["model"])
    return best_name, models[best_name], best_row


def train_evaluate_webis(
    threshold: float | None = None,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Load Webis, choose/apply a score threshold, train models, and evaluate holdout performance."""
    instances = load_webis_instances()
    truth = load_webis_truth()
    merged = merge_webis_instances_truth(instances, truth)
    threshold_report = threshold_testing(merged)

    if threshold is None:
        threshold, threshold_row = choose_webis_threshold(threshold_report)
    else:
        threshold_row = threshold_report.loc[threshold_report["threshold"] == threshold]
        threshold_row = threshold_row.iloc[0] if not threshold_row.empty else pd.Series({"threshold": threshold})

    webis = normalize_webis(merged, threshold=threshold)
    x_train, x_test, y_train, y_test = split_train_test(
        webis,
        test_size=test_size,
        random_state=random_state,
    )
    models = train_models(baseline_models(random_state=random_state), x_train, y_train)
    metrics_df, evaluation_details = evaluate_models(models, x_test, y_test)
    best_name, best_model, best_row = select_best_model(models, metrics_df, metric="macro_f1")

    return {
        "threshold": threshold,
        "threshold_row": threshold_row,
        "threshold_report": threshold_report,
        "webis": webis,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "models": models,
        "metrics": metrics_df,
        "evaluation_details": evaluation_details,
        "best_model_name": best_name,
        "best_model": best_model,
        "best_model_metrics": best_row,
    }


def test_on_external_kaggle(
    model: BaseEstimator,
    kaggle_df: pd.DataFrame,
    text_column: str = "text_clean",
    label_column: str = "label_binary",
) -> dict[str, Any]:
    """Evaluate a fitted Webis model on normalized Kaggle data."""
    return evaluate_model(model, kaggle_df[text_column], kaggle_df[label_column])


def evaluate_external_kaggle(
    model: BaseEstimator,
    kaggle_path: str | Path = KAGGLE_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load normalized Kaggle data and evaluate a fitted Webis model on it."""
    kaggle = load_normalized_kaggle(kaggle_path)
    return kaggle, test_on_external_kaggle(model, kaggle)


def threshold_testing(
    merged_webis: pd.DataFrame,
    thresholds: Iterable[float] = WEBIS_SCORE_THRESHOLDS,
) -> pd.DataFrame:
    """Summarize class balance from different Webis `truthMean` thresholds."""
    rows = []
    scores = pd.to_numeric(merged_webis["truthMean"], errors="coerce").dropna()
    for threshold in thresholds:
        labels = scores.apply(lambda score: score_to_binary(score, threshold))
        clickbait_count = int(labels.sum())
        non_clickbait_count = int((labels == 0).sum())
        minority_count = min(clickbait_count, non_clickbait_count)
        majority_count = max(clickbait_count, non_clickbait_count)
        rows.append(
            {
                "threshold": threshold,
                "n_rows": len(labels),
                "clickbait_count": clickbait_count,
                "non_clickbait_count": non_clickbait_count,
                "clickbait_rate": float(labels.mean()),
                "non_clickbait_rate": float((labels == 0).mean()),
                "minority_count": minority_count,
                "class_balance": minority_count / majority_count if majority_count else 0.0,
            }
        )
    return pd.DataFrame(rows)


def print_threshold_report(thresholds_df: pd.DataFrame) -> None:
    """Print threshold label distribution, balance, and usable rows."""
    for row in thresholds_df.itertuples(index=False):
        print(f"threshold={row.threshold}")
        print(f"  usable_rows={row.n_rows}")
        print(
            "  label_distribution="
            f"clickbait:{row.clickbait_count} ({row.clickbait_rate:.1%}), "
            f"non_clickbait:{row.non_clickbait_count} ({row.non_clickbait_rate:.1%})"
        )
        print(f"  class_balance={row.class_balance:.3f}")


def choose_webis_threshold(
    thresholds_df: pd.DataFrame,
    min_class_rate: float = 0.15,
) -> tuple[float, pd.Series]:
    """Choose a Webis score threshold with a transparent class-balance rule.

    The rule filters out thresholds where either class is too small, then picks
    the remaining threshold with the best minority/majority balance. Ties prefer
    the middle of the tested threshold range, which avoids unstable extremes.
    """
    candidates = thresholds_df[
        (thresholds_df["clickbait_rate"] >= min_class_rate)
        & (thresholds_df["non_clickbait_rate"] >= min_class_rate)
    ].copy()
    if candidates.empty:
        candidates = thresholds_df.copy()

    center = float(thresholds_df["threshold"].median())
    candidates["distance_from_center"] = (candidates["threshold"] - center).abs()
    chosen = candidates.sort_values(
        ["class_balance", "minority_count", "distance_from_center"],
        ascending=[False, False, True],
    ).iloc[0]
    return float(chosen["threshold"]), chosen


def top_features(
    model: Pipeline,
    top_n: int = 20,
) -> pd.DataFrame:
    """Extract top positive and negative TF-IDF features for linear classifiers."""
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(classifier, "coef_"):
        coefficients = classifier.coef_[0]
    elif hasattr(classifier, "feature_log_prob_"):
        coefficients = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
    else:
        raise ValueError("Top feature extraction needs coefficients or feature log probabilities.")

    ranked = pd.DataFrame({"feature": feature_names, "weight": coefficients})
    ranked["ngram_length"] = ranked["feature"].str.split().str.len()
    positive = ranked.nlargest(top_n, "weight").assign(class_label="clickbait")
    negative = ranked.nsmallest(top_n, "weight").assign(class_label="non_clickbait")
    return pd.concat([positive, negative], ignore_index=True)


def top_features_by_ngram(
    model: Pipeline,
    ngram_lengths: Iterable[int] = (1, 2, 3),
    top_n: int = 15,
) -> pd.DataFrame:
    """Extract top clickbait and non-clickbait features separately by n-gram length."""
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(classifier, "coef_"):
        coefficients = classifier.coef_[0]
    elif hasattr(classifier, "feature_log_prob_"):
        coefficients = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
    else:
        raise ValueError("N-gram feature extraction needs coefficients or feature log probabilities.")

    ranked = pd.DataFrame({"feature": feature_names, "weight": coefficients})
    ranked["ngram_length"] = ranked["feature"].str.split().str.len()

    tables = []
    for ngram_length in ngram_lengths:
        subset = ranked[ranked["ngram_length"] == ngram_length]
        if subset.empty:
            continue
        tables.append(subset.nlargest(top_n, "weight").assign(class_label="clickbait"))
        tables.append(subset.nsmallest(top_n, "weight").assign(class_label="non_clickbait"))

    if not tables:
        return pd.DataFrame(columns=["feature", "weight", "ngram_length", "class_label"])
    return pd.concat(tables, ignore_index=True)


def prediction_analysis_table(
    model: BaseEstimator,
    df: pd.DataFrame,
    text_column: str = "text_clean",
    label_column: str = "label_binary",
) -> pd.DataFrame:
    """Add predictions, errors, and available confidence values to a normalized dataset."""
    result = df.copy()
    result["predicted_label"] = model.predict(result[text_column])
    result["predicted_label_name"] = result["predicted_label"].map({0: "non_clickbait", 1: "clickbait"})
    result["true_label_name"] = result[label_column].map({0: "non_clickbait", 1: "clickbait"})
    result["is_error"] = result[label_column] != result["predicted_label"]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(result[text_column])
        result["prob_non_clickbait"] = probabilities[:, 0]
        result["prob_clickbait"] = probabilities[:, 1]
        result["predicted_probability"] = np.where(
            result["predicted_label"] == 1,
            result["prob_clickbait"],
            result["prob_non_clickbait"],
        )
        result["confidence_margin"] = (result["prob_clickbait"] - 0.5).abs()
    else:
        scores = _positive_scores(model, result[text_column])
        if scores is not None:
            result["score"] = scores

    return result


def false_positives(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Return non-clickbait examples predicted as clickbait."""
    return analysis_df[(analysis_df["label_binary"] == 0) & (analysis_df["predicted_label"] == 1)].copy()


def false_negatives(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Return clickbait examples predicted as non-clickbait."""
    return analysis_df[(analysis_df["label_binary"] == 1) & (analysis_df["predicted_label"] == 0)].copy()


def high_confidence_errors(analysis_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return the highest-confidence errors when probabilities are available."""
    if "predicted_probability" not in analysis_df.columns:
        return pd.DataFrame()
    errors = analysis_df[analysis_df["is_error"]].copy()
    return errors.sort_values("predicted_probability", ascending=False).head(n).reset_index(drop=True)


def low_confidence_predictions(analysis_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return lowest-confidence predictions when probabilities are available."""
    if "confidence_margin" not in analysis_df.columns:
        return pd.DataFrame()
    return analysis_df.sort_values("confidence_margin", ascending=True).head(n).reset_index(drop=True)


def error_analysis_table(
    model: BaseEstimator,
    df: pd.DataFrame,
    text_column: str = "text_clean",
    original_text_column: str = "text_original",
    label_column: str = "label_binary",
) -> pd.DataFrame:
    """Build a table of incorrect predictions for manual inspection."""
    result = df.copy()
    result["predicted_label"] = model.predict(result[text_column])
    scores = _positive_scores(model, result[text_column])
    if scores is not None:
        result["score"] = scores
    result["is_error"] = result[label_column] != result["predicted_label"]

    columns = [
        "dataset",
        "id",
        original_text_column,
        text_column,
        "label_raw",
        "score_raw",
        label_column,
        "predicted_label",
        "is_error",
    ]
    if "score" in result.columns:
        columns.append("score")
    return result.loc[result["is_error"], columns].reset_index(drop=True)


def plot_confusion_matrix_heatmap(
    matrix: np.ndarray,
    title: str = "Confusion matrix",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a confusion matrix heatmap from a precomputed matrix."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["non_clickbait", "clickbait"],
        yticklabels=["non_clickbait", "clickbait"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return ax


def plot_label_distribution(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot binary label counts by dataset."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="dataset", hue="label_binary", ax=ax)
    ax.set_title("Label distribution by dataset")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Count")
    ax.legend(title="Label", labels=["Non-clickbait", "Clickbait"])
    return ax


def plot_text_length_distribution(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot cleaned text length distribution by label."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    lengths = df.assign(text_length_words=df["text_clean"].str.split().str.len())
    sns.histplot(
        data=lengths,
        x="text_length_words",
        hue="label_binary",
        bins=40,
        kde=True,
        ax=ax,
    )
    ax.set_title("Cleaned text length distribution")
    ax.set_xlabel("Words")
    ax.set_ylabel("Count")
    return ax


def plot_metrics(metrics_df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot model metric comparison."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    metric_columns = ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc"]
    available = [column for column in metric_columns if column in metrics_df.columns]
    long_metrics = metrics_df.melt(id_vars="model", value_vars=available, var_name="metric", value_name="value")
    sns.barplot(data=long_metrics, x="model", y="value", hue="metric", ax=ax)
    ax.set_title("Model performance")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    return ax


def plot_confusion_matrix(
    model: BaseEstimator,
    x_test: Iterable[str],
    y_test: Iterable[int],
    ax: plt.Axes | None = None,
) -> ConfusionMatrixDisplay:
    """Plot a confusion matrix for a fitted model."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    display = ConfusionMatrixDisplay.from_estimator(
        model,
        x_test,
        y_test,
        display_labels=["non_clickbait", "clickbait"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Confusion matrix")
    return display


def plot_top_features(features_df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot feature weights returned by `top_features`."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    ordered = features_df.sort_values("weight")
    sns.barplot(data=ordered, y="feature", x="weight", hue="class_label", dodge=False, ax=ax)
    ax.set_title("Top weighted features")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Feature")
    return ax


def plot_ngram_features(
    ngram_features_df: pd.DataFrame,
    ngram_length: int,
    class_label: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot top feature weights for one n-gram length, optionally filtered by class."""
    subset = ngram_features_df[ngram_features_df["ngram_length"] == ngram_length]
    if class_label is not None:
        subset = subset[subset["class_label"] == class_label]
    if subset.empty:
        raise ValueError(f"No features found for ngram_length={ngram_length}, class_label={class_label!r}.")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    plot_top_features(subset, ax=ax)
    label = f"{ngram_length}-gram"
    if class_label is not None:
        label = f"{label} {class_label}"
    ax.set_title(f"Top {label} TF-IDF features")
    return ax


def plot_confidence_distribution(analysis_df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot predicted probability distribution when model probabilities are available."""
    if "prob_clickbait" not in analysis_df.columns:
        raise ValueError("Confidence distribution requires probability columns from predict_proba.")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(
        data=analysis_df,
        x="prob_clickbait",
        hue="true_label_name",
        bins=40,
        kde=True,
        ax=ax,
    )
    ax.set_title("Kaggle clickbait probability distribution")
    ax.set_xlabel("Predicted probability of clickbait")
    ax.set_ylabel("Count")
    return ax


def plot_threshold_comparison(thresholds_df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot class rates and class balance across tested Webis score thresholds."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    plot_df = thresholds_df.melt(
        id_vars="threshold",
        value_vars=["clickbait_rate", "non_clickbait_rate", "class_balance"],
        var_name="measure",
        value_name="value",
    )
    sns.lineplot(data=plot_df, x="threshold", y="value", hue="measure", marker="o", ax=ax)
    ax.set_title("Webis threshold comparison")
    ax.set_xlabel("TruthMean threshold")
    ax.set_ylabel("Rate / balance")
    ax.set_ylim(0, 1)
    return ax


def save_standard_figures(
    webis: pd.DataFrame,
    kaggle: pd.DataFrame,
    webis_metrics: pd.DataFrame,
    webis_confusion_matrix: np.ndarray,
    kaggle_confusion_matrix: np.ndarray,
    threshold_report: pd.DataFrame,
    features_df: pd.DataFrame,
    kaggle_analysis: pd.DataFrame,
    ngram_features_df: pd.DataFrame | None = None,
    output_dir: str | Path = PROJECT_ROOT / "outputs" / "figures",
) -> list[Path]:
    """Save the standard project visualizations and return their paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    combined = pd.concat([webis, kaggle], ignore_index=True)
    figure_specs = [
        ("label_distribution_by_dataset.png", lambda ax: plot_label_distribution(combined, ax=ax), (7, 4)),
        ("model_comparison.png", lambda ax: plot_metrics(webis_metrics, ax=ax), (9, 4)),
        (
            "webis_holdout_confusion_matrix.png",
            lambda ax: plot_confusion_matrix_heatmap(webis_confusion_matrix, "Webis holdout confusion matrix", ax=ax),
            (5, 4),
        ),
        (
            "kaggle_external_confusion_matrix.png",
            lambda ax: plot_confusion_matrix_heatmap(kaggle_confusion_matrix, "Kaggle external confusion matrix", ax=ax),
            (5, 4),
        ),
        (
            "threshold_comparison.png",
            lambda ax: plot_threshold_comparison(threshold_report, ax=ax),
            (7, 4),
        ),
        ("top_features.png", lambda ax: plot_top_features(features_df, ax=ax), (8, 8)),
        (
            "top_clickbait_features.png",
            lambda ax: plot_top_features(features_df[features_df["class_label"] == "clickbait"], ax=ax),
            (8, 6),
        ),
        (
            "top_non_clickbait_features.png",
            lambda ax: plot_top_features(features_df[features_df["class_label"] == "non_clickbait"], ax=ax),
            (8, 6),
        ),
    ]

    if ngram_features_df is not None:
        for ngram_length, label in [(1, "unigram"), (2, "bigram"), (3, "trigram")]:
            if not ngram_features_df[ngram_features_df["ngram_length"] == ngram_length].empty:
                figure_specs.append(
                    (
                        f"top_{label}_features.png",
                        lambda ax, n=ngram_length: plot_ngram_features(ngram_features_df, n, ax=ax),
                        (8, 8),
                    )
                )

        for ngram_length, label in [(2, "bigram"), (3, "trigram")]:
            for class_label in ["clickbait", "non_clickbait"]:
                subset = ngram_features_df[
                    (ngram_features_df["ngram_length"] == ngram_length)
                    & (ngram_features_df["class_label"] == class_label)
                ]
                if not subset.empty:
                    figure_specs.append(
                        (
                            f"top_{label}_{class_label}_features.png",
                            lambda ax, n=ngram_length, c=class_label: plot_ngram_features(
                                ngram_features_df,
                                n,
                                class_label=c,
                                ax=ax,
                            ),
                            (8, 6),
                        )
                    )

    if "prob_clickbait" in kaggle_analysis.columns:
        figure_specs.append(
            (
                "kaggle_confidence_distribution.png",
                lambda ax: plot_confidence_distribution(kaggle_analysis, ax=ax),
                (7, 4),
            )
        )

    for filename, plotter, size in figure_specs:
        fig, ax = plt.subplots(figsize=size)
        plotter(ax)
        path = output_path / filename
        save_figure(fig, path)
        plt.close(fig)
        saved_paths.append(path)

    return saved_paths


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    """Save a Matplotlib figure, creating parent folders if needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
