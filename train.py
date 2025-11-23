"""
Train a sentiment analysis model using:
- TF-IDF vectorization (unigrams + bigrams)
- Chi-square feature selection
- Logistic Regression classifier

This script is designed for MLOps Level 0:
- Deterministic training procedure
- Clear data preparation
- Simple artifact saving
"""

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from dataset_loader import load_dataset_amazon


# -------------------------------------------------------------------
# 1. Data loading and preparation
# -------------------------------------------------------------------

def load_and_prepare_data(
    dataset_file: str = "train.csv",
    text_full: bool = True,
    sample_size: int | None = None,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series]:
    """
    Load the Amazon reviews dataset and prepare features/labels.

    Parameters
    ----------
    dataset_file : str
        CSV file name inside the Kaggle dataset (e.g. "train.csv").
    text_full : bool
        Whether to combine title + comment into a single text_full column.
    sample_size : int or None
        If not None, randomly sample this number of rows
        (useful because the dataset is very large).
    random_state : int
        Random seed for sampling and train/test split.

    Returns
    -------
    X : pd.Series
        Text data (text_full).
    y : pd.Series
        Binary sentiment labels (0 = negative, 1 = positive).
    """
    df = load_dataset_amazon(dataset_file=dataset_file, text_full=text_full)

    # Drop rows without text
    df = df.dropna(subset=["text_full"])

    # Optional: random subsample to make training faster
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)

    # Map original labels (1 = negative, 2 = positive) to 0/1
    label_map = {1: 0, 2: 1}
    df["label_binary"] = df["label"].map(label_map)

    # Keep only rows with valid labels after mapping
    df = df.dropna(subset=["label_binary"])

    X = df["text_full"]
    y = df["label_binary"].astype(int)

    return X, y


# -------------------------------------------------------------------
# 2. Build the sklearn Pipeline (TF-IDF + chi2 + Logistic Regression)
# -------------------------------------------------------------------

def build_pipeline(
    max_features: int = 50000,
    k_best: int = 10000,
) -> Pipeline:
    """
    Build a scikit-learn Pipeline:
    - TF-IDF vectorization
    - Chi-square feature selection
    - Logistic Regression classifier

    Parameters
    ----------
    max_features : int
        Maximum number of TF-IDF features to keep.
    k_best : int
        Number of features to select using chi-square.

    Returns
    -------
    pipeline : Pipeline
        Configured scikit-learn Pipeline object.
    """
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),    # unigrams + bigrams
        stop_words="english",
        max_features=max_features,
        min_df=5,              # ignore very rare terms
    )

    selector = SelectKBest(score_func=chi2, k=k_best)

    clf = LogisticRegression(
        solver="liblinear",    # stable for smaller datasets and sparse data
        max_iter=1000,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", tfidf),
            ("chi2", selector),
            ("clf", clf),
        ]
    )

    return pipeline


# -------------------------------------------------------------------
# 3. Training and evaluation
# -------------------------------------------------------------------

def train_and_evaluate(
    dataset_file: str = "train.csv",
    sample_size: int | None = 50000,  # adjust based on hardware
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 50000,
    k_best: int = 10000,
) -> dict:
    """
    Load data, train the pipeline, evaluate on a hold-out set,
    and return the fitted model and metrics.

    Parameters
    ----------
    dataset_file : str
        Name of the CSV file to load from the Kaggle dataset.
    sample_size : int or None
        If not None, randomly sample this number of rows for training.
    test_size : float
        Proportion of data reserved for testing.
    random_state : int
        Random seed for reproducibility.
    max_features : int
        Max number of TF-IDF features.
    k_best : int
        Number of features to keep after chi-square selection.

    Returns
    -------
    result : dict
        Dictionary with:
        - "model": fitted Pipeline
        - "metrics": dict with accuracy, AUC, etc.
        - "y_test": ground truth labels for the test set
        - "y_pred": predicted labels
        - "y_proba": predicted probabilities
    """
    # Load and prepare data
    X, y = load_and_prepare_data(
        dataset_file=dataset_file,
        text_full=True,
        sample_size=sample_size,
        random_state=random_state,
    )

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Build pipeline
    pipeline = build_pipeline(
        max_features=max_features,
        k_best=k_best,
    )

    # Fit pipeline
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": acc,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    print("=== Evaluation Metrics ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC:  {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        "model": pipeline,
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


# -------------------------------------------------------------------
# 4. Save artifacts (MLOps Level 0 style)
# -------------------------------------------------------------------

def save_artifacts(model: Pipeline, metrics: dict, output_dir: str = "artifacts") -> None:
    """
    Save the trained model and metrics to disk.

    Parameters
    ----------
    model : Pipeline
        Trained sklearn Pipeline (TF-IDF + chi2 + LogisticRegression).
    metrics : dict
        Dictionary with evaluation metrics.
    output_dir : str
        Directory where artifacts will be stored.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model as a single object (includes vectorizer + selector + classifier)
    model_path = output_path / "sentiment_pipeline.joblib"
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path.resolve()}")

    # Save metrics as JSON
    import json
    metrics_path = output_path / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path.resolve()}")


# -------------------------------------------------------------------
# 5. Main entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    # You can tweak these hyperparameters for experiments
    result = train_and_evaluate(
        dataset_file="train.csv",     # change if your file has a different name
        sample_size=50000,            # None = use full dataset (may be very slow)
        test_size=0.2,
        random_state=42,
        max_features=50000,
        k_best=10000,
    )

    save_artifacts(
        model=result["model"],
        metrics=result["metrics"],
        output_dir="artifacts",
    )
