import argparse
from pathlib import Path

import joblib
import numpy as np

from dataset_loader import clean_text


def load_model(model_path: str = "models/model.pkl"):
    """
    Load a trained model from disk.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. "
            "Run first: python train.py"
        )
    return joblib.load(path)


def main():
    parser = argparse.ArgumentParser(description="Sentiment inference")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text review to classify.",
    )
    args = parser.parse_args()

    model = load_model()

    raw_text = args.text
    processed_text = clean_text(raw_text)

    probs = model.predict_proba([processed_text])[0]
    classes = model.classes_

    idx = int(np.argmax(probs))

    # Prediction
    prediction = "Negative" if classes[idx] == 1 else "Positive"

    pred_label = classes[idx]
    pred_score = float(probs[idx])

    print("Input text (raw):")
    print(raw_text)
    print("\nInput text (processed):")
    print(processed_text)
    print(f"\nPrediction: {prediction}")
    print(f"Label: {pred_label}")
    print(f"Score: {pred_score:.4f}")


if __name__ == "__main__":
    main()
