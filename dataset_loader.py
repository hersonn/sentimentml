import os
from pathlib import Path

import kagglehub
import pandas as pd


# Download a Kaggle dataset to the local cache and return its version path.
def download_dataset(dataset: str) -> str:
    cache_root = Path.cwd()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(cache_root)

    dataset_root = cache_root / "datasets" / dataset.replace("/", os.sep)
    versions_dir = dataset_root / "versions"

    # If we already have at least one cached version, reuse the latest.
    if versions_dir.exists():
        versions = sorted([v for v in versions_dir.iterdir() if v.is_dir()])
        if versions:
            latest_version = versions[-1]
            print(f"Dataset already downloaded. Reusing cached version at {latest_version}")
            return str(latest_version)

    # Download dataset (kagglehub will store it under the cache directory).
    path = kagglehub.dataset_download(dataset)
    print("Path to dataset files:", path)
    return path


def load_dataset_amazon(file: str) -> pd.DataFrame:

    # Kaggle dataset
    dataset_name = "kritanjalijain/amazon-reviews"
    dataset_file = file

    # Download dataset
    path = download_dataset(dataset_name)

    # Load dataset (ignoring overflow errors)
    dataset = pd.read_csv(
        os.path.join(path, dataset_file),
        engine="python",
        on_bad_lines="warn",
        encoding="utf-8",
        header=None,
        names=["label", "title", "comment"]
    )

    return dataset


def load_dataset_amazon_test() -> pd.DataFrame:
    return load_dataset_amazon("test.csv")


def load_dataset_amazon_train() -> pd.DataFrame:
    return load_dataset_amazon("train.csv")

