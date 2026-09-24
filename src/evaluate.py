"""Evaluate the trained model on the held-out test set.

Produces, under reports/:

    metrics/
        test_metrics.json          overall loss / accuracy
        classification_report.csv  per-class precision, recall, F1, support
        confusion_matrix.csv       raw NxN counts
        per_class_accuracy.csv     correct / total images per class
    figures/
        confusion_matrix.png       annotated heatmap
        per_class_accuracy.png     bar chart
        sample_predictions.png     one confidently-correct prediction per class
        misclassified_samples.png  top mistakes by wrong-class confidence

Run from the project root:

    python src/evaluate.py
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from .config import (
    BATCH_SIZE,
    CLASS_NAMES_PATH,
    DATA_DIR,
    FIGURES_DIR,
    IMG_SIZE,
    METRICS_DIR,
    MODEL_PATH,
    load_class_names,
)
from .models import build_vegetable_cnn, compile_vegetable_cnn
from .predict import get_model

N_CORRECT_SAMPLES = 15   # one per class (5x3 grid)
N_MISCLASSIFIED = 16     # 4x4 grid

sns.set_theme(style="whitegrid")


def _build_test_dataset():
    """tf.data pipeline over the held-out test split.

    Raw pixel values in [0, 255] (matches how the model was trained).
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test",
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=False,
    )
    class_names = ds.class_names
    return ds.prefetch(tf.data.AUTOTUNE), class_names


def _collect_predictions(model, class_names):
    """One deterministic pass over the test set.

    Returns true/predicted labels, per-image probabilities, one confidently
    correct sample image per class, and the worst mistakes. Memory-friendly:
    only the selected images are retained, stored as uint8.
    """
    test_ds, ds_names = _build_test_dataset()
    assert ds_names == class_names, "Test-split ordering differs from class_names."

    y_true, y_pred = [], []
    correct_samples = {}   # class_index -> (image_uint8, prob)
    mistakes = []          # {"prob", "image", "true", "pred"}

    for x, y in test_ds:
        probs = model(x, training=False).numpy()
        yt = np.argmax(y.numpy(), axis=1)
        yp = np.argmax(probs, axis=1)

        y_true.extend(yt.tolist())
        y_pred.extend(yp.tolist())

        for img, true_idx, pred_idx, row in zip(x.numpy(), yt, yp, probs):
            img_u8 = np.clip(img, 0.0, 255.0).astype(np.uint8)
            if true_idx == pred_idx and true_idx not in correct_samples:
                correct_samples[true_idx] = (img_u8, float(row[pred_idx]))
            elif true_idx != pred_idx:
                mistakes.append(
                    {
                        "prob": float(row[pred_idx]),
                        "image": img_u8,
                        "true": int(true_idx),
                        "pred": int(pred_idx),
                    }
                )

    mistakes.sort(key=lambda d: d["prob"], reverse=True)
    return (
        np.asarray(y_true),
        np.asarray(y_pred),
        correct_samples,
        mistakes,
    )


def _compute_loss_accuracy(model):
    """Recompile a weight copy of the model to also report test loss."""
    test_ds, _ = _build_test_dataset()
    clone = compile_vegetable_cnn(build_vegetable_cnn())
    clone.set_weights(model.get_weights())
    try:
        return clone.evaluate(test_ds, verbose=0, return_dict=True)
    except Exception:
        return {}


def save_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar_kws={"shrink": 0.75},
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix - test set (n = {cm.sum():,})")
    plt.tight_layout()
    out = FIGURES_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def save_per_class_accuracy(per_class, class_names):
    fig, ax = plt.subplots(figsize=(10, 6))
    df = per_class.set_index("class")
    colors = ["#55A868" if acc >= 0.9 else "#C44E52" for acc in df["accuracy"]]
    ax.bar(df.index, df["accuracy"] * 100, color=colors)
    ax.axhline(90, color="#999999", ls="--", lw=1.2, label="90% reference")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 102)
    ax.set_title("Per-class accuracy on the test set")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    plt.tight_layout()
    out = FIGURES_DIR / "per_class_accuracy.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def save_sample_predictions(correct_samples, class_names):
    rows, cols = 3, 5
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
    for i in range(rows * cols):
        ax = axes.flat[i]
        if i in correct_samples:
            img, prob = correct_samples[i]
            ax.imshow(img)
            ax.set_title(f"{class_names[i]}\n{prob * 100:.1f}% confident", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("A confidently-correct prediction for each class (test set)", fontsize=12)
    plt.tight_layout()
    out = FIGURES_DIR / "sample_predictions.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def save_misclassified(top, class_names):
    if not top:
        return None
    take = top[:N_MISCLASSIFIED]
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    for i, d in enumerate(take):
        ax = axes.flat[i]
        ax.imshow(d["image"])
        ax.set_title(
            f"true {class_names[d['true']]}\n"
            f"pred {class_names[d['pred']]} ({d['prob'] * 100:.0f}%)",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    for i in range(len(take), rows * cols):
        axes.flat[i].axis("off")
    fig.suptitle(
        f"Top {len(take)} misclassified (by confidence in the wrong class)", fontsize=12
    )
    plt.tight_layout()
    out = FIGURES_DIR / "misclassified_samples.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def evaluate_model(
    model_path=MODEL_PATH,
    class_names_path=CLASS_NAMES_PATH,
) -> dict:
    """Run the full evaluation pipeline and persist all reports."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model = get_model(model_path)
    class_names = load_class_names(class_names_path)

    y_true, y_pred, correct_samples, mistakes = _collect_predictions(model, class_names)

    test_metrics = _compute_loss_accuracy(model)

    metrics = {
        "test_loss": float(test_metrics.get("loss", "nan"))
        if "loss" in test_metrics else None,
        "test_accuracy": float(test_metrics.get("accuracy", np.nan)),
        "manual_accuracy": float(np.mean(y_true == y_pred)),
        "n_samples": int(len(y_true)),
        "misclassified": int(np.sum(y_true != y_pred)),
        "misclassification_rate": float(np.mean(y_true != y_pred)),
        "input_size": list(IMG_SIZE),
        "num_classes": len(class_names),
    }
    (METRICS_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )
    report_df = pd.DataFrame(report).T
    report_df.to_csv(METRICS_DIR / "classification_report.csv")

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        METRICS_DIR / "confusion_matrix.csv"
    )

    per_class = pd.DataFrame(
        {
            "class": class_names,
            "accuracy": [cm[i, i] / cm[i].sum() for i in range(len(class_names))],
            "correct": [int(cm[i, i]) for i in range(len(class_names))],
            "total": [int(cm[i].sum()) for i in range(len(class_names))],
        }
    )
    per_class.to_csv(METRICS_DIR / "per_class_accuracy.csv", index=False)

    # --- figures ------------------------------------------------------- #
    save_confusion_matrix(cm, class_names)
    save_per_class_accuracy(per_class, class_names)
    save_sample_predictions(correct_samples, class_names)
    save_misclassified(mistakes, class_names)

    return {
        "metrics": metrics,
        "report": report_df,
        "confusion_matrix": cm,
        "per_class": per_class,
        "misclassified_max_shown": len(mistakes[:N_MISCLASSIFIED]),
    }


def main() -> None:
    result = evaluate_model()
    m = result["metrics"]
    print("=" * 62)
    print("TEST-SET EVALUATION  (model: models/vegetable_cnn.h5)")
    print("=" * 62)
    print(f"  Accuracy            : {m['test_accuracy'] * 100:.2f}%")
    print(f"  Misclassified       : {m['misclassified']} / {m['n_samples']}")
    print(f"  Samples             : {m['n_samples']}")
    print(f"  Input size          : {m['input_size'][0]}x{m['input_size'][1]}")
    print()
    print("  Reports written to reports/")
    print(f"    - metrics : {METRICS_DIR}")
    print(f"    - figures : {FIGURES_DIR}")


if __name__ == "__main__":
    main()