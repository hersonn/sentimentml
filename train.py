from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from dataset_loader import load_dataset_amazon
from tmp import create_tmp


def load_data():
    """
    Load dataset, apply text preprocessing and split into
    training and validation sets.
    """
    # Ensure tmp directory exists and is clean
    create_tmp()

    # Select dataset file
    dataset_file = "test.csv"
    # dataset_file = "train.csv"

    # Load dataset with cleaned text
    df = load_dataset_amazon(
        dataset_file=dataset_file,
        text_full=True,
        text_clean=True,
        with_eda_features=False,
        # sample_size=50000,  # limit the dataset for memory purposes
    )
    print(f"Loaded dataset {dataset_file} with shape: {df.shape}")

    # Keep only the columns required for training
    if "text_clean" not in df.columns:
        raise ValueError("Column 'text_clean' not found in dataframe.")
    if "label" not in df.columns:
        raise ValueError("Column 'label' not found in dataframe.")

    X = df["text_clean"]
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_val, y_train, y_val


def create_model() -> Pipeline:
    """
    Create a TF-IDF + Logistic Regression pipeline.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=50_000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    print("Loading data...")
    X_train, X_val, y_train, y_val = load_data()

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    print("\nCreating model...")
    model = create_model()

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating on validation set...")
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")
    cm = confusion_matrix(y_val, y_pred)

    print("\n===== Validation Metrics =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (weighted): {f1:.4f}")
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_val, y_pred))

    # Save trained model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "model.pkl"
    joblib.dump(model, model_path)

    print(f"\nModel saved at: {model_path.resolve()}")


if __name__ == "__main__":
    main()
