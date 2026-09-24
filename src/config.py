"""Central project configuration: paths, constants and shared settings.

Every module and notebook in the repo should derive its settings from here so
that paths and hyper-parameters stay consistent across the pipeline.
"""

import json
from pathlib import Path

# --------------------------------------------------------------------------- #
# Project layout
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "vegetable_cnn.h5"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

ASSETS_DIR = PROJECT_ROOT / "assets"
SAMPLES_DIR = ASSETS_DIR / "samples"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# --------------------------------------------------------------------------- #
# Model / training constants (match the shipped model in models/)
# --------------------------------------------------------------------------- #

SPLITS = ("train", "validation", "test")

# Input size of the trained model saved in models/vegetable_cnn.h5
IMG_SIZE = (150, 150)

BATCH_SIZE = 32
SEED = 42
NUM_CLASSES = 15

# Class names, in the exact order the model outputs them
# (alphabetical, matching tf.keras.utils.image_dataset_from_directory).
DEFAULT_CLASS_NAMES = [
    "Bean",
    "Bitter_Gourd",
    "Bottle_Gourd",
    "Brinjal",
    "Broccoli",
    "Cabbage",
    "Capsicum",
    "Carrot",
    "Cauliflower",
    "Cucumber",
    "Papaya",
    "Potato",
    "Pumpkin",
    "Radish",
    "Tomato",
]

KAGGLE_DATASET = "misrakahmed/vegetable-image-dataset"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def ensure_dirs() -> None:
    """Create every output directory that the pipeline writes to."""
    for directory in (
        DATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        ASSETS_DIR,
        SAMPLES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def load_class_names(path: "Path | str" = CLASS_NAMES_PATH) -> "list[str]":
    """Load the ordered class-name list.

    Prefers models/class_names.json; falls back to DEFAULT_CLASS_NAMES when the
    file is missing so the pipeline stays runnable without a trained model.
    """
    path = Path(path)
    if path.is_file():
        mapping = json.loads(path.read_text())
        if isinstance(mapping, dict):
            return [mapping[str(i)] for i in sorted(int(k) for k in mapping)]
        return list(mapping)
    return list(DEFAULT_CLASS_NAMES)


ensure_dirs()