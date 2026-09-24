"""Classify a single vegetable image with the trained CNN.

Preprocessing mirrors the inference path used during training
(notebooks/02_preprocessing.ipynb and 04_train_on_kaggle.ipynb):
RGB image resized to the model's input size, raw pixel values kept in
[0, 255] (the model was trained on raw pixels; no augmentation at inference).

Loads:
    - the trained model from models/vegetable_cnn.h5
    - the class-name list from models/class_names.json

Usage:
    python src/predict.py path/to/image.jpg
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

import tensorflow as tf

from .config import (
    CLASS_NAMES_PATH,
    DEFAULT_CLASS_NAMES,
    IMG_SIZE,
    MODEL_PATH,
    load_class_names,
)

# Cached singletons so the Streamlit app can reuse the loaded model across
# reruns without paying the loading cost every time.
_model: Optional[tf.keras.Model] = None
_class_names: Optional[List[str]] = None


def get_class_names(
    class_names_path: Union[str, Path] = CLASS_NAMES_PATH,
) -> List[str]:
    """Load the class-name list, in the same order the model was trained on.

    Parsing lives in `src.config.load_class_names` (single source of truth);
    this just caches the result and keeps the CLI/app log lines.
    """
    global _class_names
    if _class_names is not None:
        return _class_names

    path = Path(class_names_path)
    if path.is_file():
        print(f"Class names loaded from {path}")
    else:
        print(f"WARNING: {path} not found; using default {len(DEFAULT_CLASS_NAMES)} classes")

    _class_names = load_class_names(path)
    return _class_names


def get_model(model_path: Union[str, Path] = MODEL_PATH) -> tf.keras.Model:
    """Load the trained model once and cache it for reuse."""
    global _model
    if _model is None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Model not found at {path}. Download the trained model "
                "(e.g. from Kaggle) into models/."
            )
        print(f"Loading model from {path} ...")
        _model = tf.keras.models.load_model(path)
        print("Model loaded.")
    return _model


def get_input_size(model: tf.keras.Model) -> tuple:
    """Return the (height, width) the model was trained with.

    Reads the model's declared input shape so inference matches training even
    when the saved model was built with a non-default shape.
    """
    shape = model.inputs[0].shape  # e.g. (None, 150, 150, 3)
    height, width = shape[1], shape[2]
    if height is not None and width is not None:
        return (int(height), int(width))
    return IMG_SIZE


def preprocess_image(
    image_path: Union[str, Path],
    target_size: Optional[tuple] = None,
) -> "tf.Tensor":
    """Load one image the same way training data was prepared.

    Args:
        image_path: path to the image file.
        target_size: (height, width) to resize to; defaults to IMG_SIZE.

    Returns a float32 tensor of shape (1, height, width, 3) with pixel values
    in [0, 255]. The saved model was trained on raw pixels (see
    notebooks/04_train_on_kaggle.ipynb), so we do NOT normalize to [0, 1].
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    if target_size is None:
        target_size = IMG_SIZE
    img = tf.keras.utils.load_img(path, target_size=target_size, color_mode="rgb")
    return tf.keras.utils.img_to_array(img)[None, ...]  # (1, h, w, 3) float32


def predict_top_k(
    image_path: Union[str, Path],
    model: Optional[tf.keras.Model] = None,
    class_names: Optional[List[str]] = None,
    k: int = 5,
) -> Dict[str, object]:
    """Predict the top-k classes of one image.

    Returns a dict with predict() keys plus `top_k`, a list of
    (class_name, probability) tuples, most likely first.

    Raises before predicting if the class list and the model disagree, so a
    stale `class_names.json` can never silently produce wrong labels.
    """
    if model is None:
        model = get_model()
    if class_names is None:
        class_names = get_class_names()

    n_out = model.output_shape[-1] if model.output_shape is not None else None
    if n_out is not None and len(class_names) != n_out:
        raise ValueError(
            f"Label mismatch: the model outputs {n_out} classes, but "
            f"class_names contains {len(class_names)}. Update "
            "models/class_names.json to match the trained model."
        )

    target_size = get_input_size(model)
    probs = model(
        preprocess_image(image_path, target_size=target_size), training=False
    ).numpy()[0]

    order = probs.argsort()[::-1][:k]
    top_k = [(class_names[i], float(probs[i])) for i in order]
    return {
        "class_index": int(order[0]),
        "class_name": class_names[order[0]],
        "confidence": float(probs[order[0]]),
        "top_k": top_k,
    }


def predict(
    image_path: Union[str, Path],
    model: Optional[tf.keras.Model] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Predict the class of a single image (top-1, as before)."""
    result = predict_top_k(image_path, model=model, class_names=class_names, k=1)
    return {
        k: result[k] for k in ("class_index", "class_name", "confidence")
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify a single image with the trained vegetable CNN."
    )
    parser.add_argument("image", type=Path, help="Path to an image file")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help=f"Path to the saved model (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--class-names",
        type=Path,
        default=CLASS_NAMES_PATH,
        help=f"Path to the class-names JSON (default: {CLASS_NAMES_PATH})",
    )
    args = parser.parse_args()

    class_names = get_class_names(args.class_names)
    model = get_model(args.model)
    result = predict(args.image, model=model, class_names=class_names)

    print(f"\nPrediction: {result['class_name']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    print(f"Class index: {result['class_index']} / {len(class_names) - 1}")


if __name__ == "__main__":
    main()