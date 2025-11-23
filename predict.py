"""
Simple inference script for the sentiment analysis model.

- Loads the trained sklearn Pipeline (TF-IDF + chi2 + LogisticRegression)
- Provides a function-level API (predict_text)
- Provides a CLI interface for quick testing

Usage (from project root):

    python src/models/predict.py --text "this product is amazing"

Assumptions:
- The trained model artifact is stored at: artifacts/sentiment_pipeline.joblib
- The model outputs probabilities for class 1 ("positive")
"""

import argparse
from pathlib import Path
from typing import Dict, Union, List

import joblib


# -------------------------------------------------------------------
# 1. Model loading utilities
# -------------------------------------------------------------------

def load_model(model_path: Union[str, Path] = "artifacts/sentiment_pipeline.joblib"):
    """
    Load the trained sentiment analysis pipeline from disk.

    Parameters
    ----------
    model_path : str or Path
        Path to the joblib file containing the trained Pipeline.

    Returns
    -------
    model :
        Loaded sklearn Pipeline object.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            f"Train the model first (train.py) and ensure artifacts are in place."
        )
    model = joblib.load(model_path)
    return model


# Load once at module import time (good enough for simple scripts / Level 0)
MODEL = None


def get_model():
    """
    Lazy-load global model instance.

    This avoids reloading the joblib artifact on every prediction.
    """
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    return MODEL


# -------------------------------------------------------------------
# 2. Core prediction logic
# -------------------------------------------------------------------

def predict_text(text: str) -> Dict[str, Union[str, float]]:
    """
    Predict sentiment for a single raw text string.

    Parameters
    ----------
    text : str
        Raw review text.

    Returns
    -------
    result : dict
        Dictionary with:
        - "label": "negative" or "positive"
        - "score": probability of the positive class (float in [0, 1])
    """
    model = get_model()

    # Model expects an iterable of texts, so wrap it in a list
    proba = model.predict_proba([text])[0]  # shape: (2,)
    # By construction: proba[0] = P(class=0="negative"), proba[1] = P(class=1="positive")
    prob_negative = float(proba[0])
    prob_positive = float(proba[1])

    # Choose the most likely class
    if prob_positive >= prob_negative:
        label = "positive"
        score = prob_positive
    else:
        label = "negative"
        score = prob_negative

    return {
        "label": label,
        "score": score,
    }


def predict_batch(texts: List[str]) -> List[Dict[str, Union[str, float]]]:
    """
    Predict sentiment for a batch of texts.

    Parameters
    ----------
    texts : list of str
        List of raw review texts.

    Returns
    -------
    results : list of dict
        Each element is a dict like the one returned by predict_text.
    """
    model = get_model()
    proba = model.predict_proba(texts)  # shape: (n_samples, 2)

    results = []
    for i, text in enumerate(texts):
        prob_negative = float(proba[i, 0])
        prob_positive = float(proba[i, 1])

        if prob_positive >= prob_negative:
            label = "positive"
            score = prob_positive
        else:
            label = "negative"
            score = prob_negative

        results.append(
            {
                "text": text,
                "label": label,
                "score": score,
            }
        )

    return results


# -------------------------------------------------------------------
# 3. CLI interface
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    You can either:
    - Pass a single text with --text
    - (Optional) Extend later to support --file for batch inference
    """
    parser = argparse.ArgumentParser(
        description="Sentiment prediction for Amazon product reviews."
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Raw review text to classify.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/sentiment_pipeline.joblib",
        help="Path to the trained model artifact (joblib).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Override global model path if custom path is provided
    global MODEL
    MODEL = load_model(args.model_path)

    result = predict_text(args.text)

    print("Input text:")
    print(args.text)
    print("\nPrediction:")
    print(f"Label: {result['label']}")
    print(f"Score: {result['score']:.4f}")


if __name__ == "__main__":
    main()
