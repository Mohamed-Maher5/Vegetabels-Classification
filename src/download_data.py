"""Download the Vegetable Image Dataset from Kaggle and unpack it into data/.

Normalizes the archive (which ships as ``Vegetable Images/train|validation|test``)
into the layout the rest of the pipeline expects::

    data/
        train/<class>/image.jpg
        validation/<class>/image.jpg
        test/<class>/image.jpg

Run::

    python src/download_data.py

Requires Kaggle credentials (set KAGGLE_USERNAME / KAGGLE_KEY env vars or
create ~/.kaggle/kaggle.json).
"""

import os
import shutil
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from .config import DATA_DIR, KAGGLE_DATASET, RAW_DATA_DIR, SPLITS


def check_credentials() -> None:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    access_token = kaggle_dir / "access_token"
    has_env = "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ
    has_json = kaggle_json.exists()
    has_token = access_token.exists() or "KAGGLE_API_TOKEN" in os.environ
    if not (has_env or has_json or has_token):
        raise RuntimeError(
            "Kaggle credentials not found. Authenticate by running "
            "`kaggle auth login`, or set the KAGGLE_USERNAME and KAGGLE_KEY "
            "environment variables, or create ~/.kaggle/kaggle.json "
            "containing your username and key."
        )


def download_dataset() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{KAGGLE_DATASET}' via the Kaggle API ...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DATA_DIR), quiet=False)
    print(f"Downloaded archives into {RAW_DATA_DIR}")


def extract_archives() -> None:
    for archive in sorted(RAW_DATA_DIR.glob("*.zip")):
        dest = RAW_DATA_DIR / archive.stem
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {archive.name} -> {dest}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        while True:
            nested = list(dest.rglob("*.zip"))
            if not nested:
                break
            for nz in nested:
                with zipfile.ZipFile(nz) as nzf:
                    nzf.extractall(dest)
                nz.unlink()


def normalize_splits() -> None:
    for split in SPLITS:
        target = DATA_DIR / split
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)

    dirs = [p for p in RAW_DATA_DIR.rglob("*") if p.is_dir()]
    found_any = False
    for split in SPLITS:
        hits = [d for d in dirs if d.name.lower() == split]
        if not hits:
            print(f"WARNING: no '{split}' folder found under {RAW_DATA_DIR}")
            continue
        found_any = True
        src = hits[0]
        for class_dir in sorted(p for p in src.iterdir() if p.is_dir()):
            shutil.copytree(class_dir, DATA_DIR / split / class_dir.name)
    if not found_any:
        raise FileNotFoundError(
            f"Could not find train/validation/test folders under {RAW_DATA_DIR}."
        )


def verify_structure() -> None:
    print("\n=== Data structure ===")
    for split in SPLITS:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            print(f"{split}: MISSING")
            continue
        classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        counts = {name: len(list((split_dir / name).glob("*"))) for name in classes}
        total = sum(counts.values())
        print(f"{split}: {len(classes)} classes, {total} images")
        print(f"  classes ({len(classes)}): {', '.join(classes)}")
    print(f"\nDataset root: {DATA_DIR.resolve()}")


def main() -> None:
    check_credentials()
    download_dataset()
    extract_archives()
    normalize_splits()
    verify_structure()


if __name__ == "__main__":
    main()