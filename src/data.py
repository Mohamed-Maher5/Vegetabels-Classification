"""Dataset helpers: scan the data/ folder into a tidy data frame.

Used by notebooks for EDA and by the reporting scripts to summarise the data.
"""

import pandas as pd

from .config import DATA_DIR, SPLITS

_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_dataset_frame(splits: "tuple[str, ...]" = SPLITS) -> pd.DataFrame:
    """Return one row per image: split, class and full file path."""
    rows = []
    for split in splits:
        split_dir = DATA_DIR / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for path in class_dir.rglob("*"):
                if path.suffix.lower() in _EXTENSIONS:
                    rows.append({"split": split, "class": class_dir.name, "path": path})

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(
            f"No images found under {DATA_DIR}. Run `python src/download_data.py` "
            "first (requires Kaggle credentials)."
        )
    return frame


def class_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Images per class per split + a class total column."""
    summary = (
        frame
        .pivot_table(index="class", columns="split", values="path", aggfunc="count")
        .reindex(columns=list(SPLITS))
        .fillna(0)
        .astype(int)
    )
    summary["total"] = summary.sum(axis=1)
    return summary.sort_index()


def dataset_stats(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per split: number of classes, images, first/last class name."""
    stats = []
    for split in SPLITS:
        sub = frame[frame["split"] == split]
        if sub.empty:
            continue
        classes = sorted(sub["class"].unique())
        stats.append(
            {
                "split": split,
                "classes": len(classes),
                "images": len(sub),
                "images_per_class": int(round(len(sub) / len(classes))),
            }
        )
    return pd.DataFrame(stats)