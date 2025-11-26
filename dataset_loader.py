import os
import re
from pathlib import Path
from typing import Optional

import kagglehub
import pandas as pd


def download_dataset(dataset: str) -> str:
    """
    Download a Kaggle dataset to the local cache and return its version path.
    If a cached version already exists, reuse the latest one.
    """
    cache_root = Path.cwd()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(cache_root)

    dataset_root = cache_root / "datasets" / dataset.replace("/", os.sep)
    versions_dir = dataset_root / "versions"

    # Reuse cached version if available
    if versions_dir.exists():
        versions = sorted([v for v in versions_dir.iterdir() if v.is_dir()])
        if versions:
            latest_version = versions[-1]
            print(f"Dataset already downloaded. Reusing cached version at {latest_version}")
            return str(latest_version)

    # Download dataset (kagglehub will store it under the cache directory)
    path = kagglehub.dataset_download(dataset)
    print("Path to dataset files:", path)
    return path


def clean_text(text: str) -> str:
    """
    Clean and normalize raw text:
    - lowercase
    - remove HTML tags
    - remove URLs
    - keep only alphanumeric characters and spaces
    - collapse multiple spaces
    """
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_eda_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Add auxiliary columns for exploratory data analysis (EDA).
    These features are not used directly in model training.
    """
    # Length-based features
    dataset["num_chars"] = dataset["text_clean"].str.len()
    dataset["num_words"] = dataset["text_clean"].str.split().apply(len)
    dataset["avg_word_length"] = dataset["text_clean"].apply(
        lambda x: sum(len(w) for w in x.split()) / max(len(x.split()), 1)
    )

    # Vocabulary diversity
    dataset["num_unique_words"] = dataset["text_clean"].apply(
        lambda x: len(set(x.split()))
    )
    dataset["unique_ratio"] = (
        dataset["num_unique_words"] / dataset["num_words"].replace(0, 1)
    )

    # Digits and punctuation
    dataset["num_digits"] = dataset["text_clean"].str.count(r"\d")
    dataset["num_punct"] = dataset["text_full"].str.count(r"[^\w\s]")

    # URLs and uppercase ratio
    dataset["has_url"] = dataset["text_full"].str.contains(r"http[s]?://", regex=True)
    dataset["has_uppercase_ratio"] = dataset["text_full"].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )

    return dataset


def load_dataset_amazon(
    dataset_file: str,
    text_full: bool,
    text_clean: bool,
    with_eda_features: bool = False,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load Amazon reviews dataset from Kaggle, optionally:
    - combine title + comment into `text_full`
    - clean text into `text_clean`
    - add EDA features (for analysis, not training)
    - optionally subsample the dataset for memory constraints
    """
    dataset_name = "kritanjalijain/amazon-reviews"

    # Download dataset
    path = download_dataset(dataset_name)

    # Load dataset (ignoring overflow errors)
    dataset = pd.read_csv(
        os.path.join(path, dataset_file),
        engine="python",
        on_bad_lines="warn",
        encoding="utf-8",
        header=None,
        names=["label", "title", "comment"],
    )

    # Optional subsample for memory / faster experiments
    if sample_size is not None and sample_size < len(dataset):
        dataset = dataset.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Combine title and comment into text_full
    if text_full:
        dataset["text_full"] = (
            dataset["title"].astype(str) + " " + dataset["comment"].astype(str)
        ).str.strip()

    # Clean text
    if text_clean:
        if "text_full" not in dataset.columns:
            # Fallback in case text_full=False but we still want clean_text
            dataset["text_full"] = (
                dataset["title"].astype(str) + " " + dataset["comment"].astype(str)
            ).str.strip()
        dataset["text_clean"] = dataset["text_full"].astype(str).apply(clean_text)

    # Add EDA features only when requested
    if with_eda_features:
        if "text_clean" not in dataset.columns:
            raise ValueError("text_clean must be True to compute EDA features.")
        dataset = add_eda_features(dataset)

    return dataset
