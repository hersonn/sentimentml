import shutil
from pathlib import Path

TMP_DIR = Path("tmp")


def create_tmp() -> None:
    """
    Delete tmp directory if it exists and recreate it from scratch.
    """
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Recreated directory: {TMP_DIR.resolve()}")


def delete_tmp() -> None:
    """
    Delete tmp directory if it exists.
    """
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
        print(f"Deleted directory: {TMP_DIR.resolve()}")
