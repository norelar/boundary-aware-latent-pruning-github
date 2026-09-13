"""Classification metrics, calibration, and confusion-matrix outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 15,
) -> float:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correctness = (predictions == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    value = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence > lower) & (confidence <= upper)
        if index == 0:
            mask |= confidence == 0.0
        if not mask.any():
            continue
        value += float(mask.mean()) * abs(float(correctness[mask].mean()) - float(confidence[mask].mean()))
    return float(value)


def classification_metrics(
    y_true: Sequence[int],
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> tuple[dict[str, float | None], pd.DataFrame, np.ndarray]:
    truth = np.asarray(y_true, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    if probability.ndim != 2 or probability.shape[0] != len(truth):
        raise ValueError("Probability matrix must be [number of samples, number of classes]")
    predictions = probability.argmax(axis=1)
    labels = np.arange(len(class_names))
    macro = precision_recall_fscore_support(
        truth, predictions, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        truth, predictions, labels=labels, average="weighted", zero_division=0
    )
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "mcc": float(matthews_corrcoef(truth, predictions)),
        "nll": float(log_loss(truth, probability, labels=labels)),
        "ece": expected_calibration_error(truth, probability),
        "roc_auc_macro_ovr": None,
    }
    try:
        if len(class_names) == 2:
            metrics["roc_auc_macro_ovr"] = float(roc_auc_score(truth, probability[:, 1]))
        else:
            metrics["roc_auc_macro_ovr"] = float(
                roc_auc_score(truth, probability, labels=labels, average="macro", multi_class="ovr")
            )
    except ValueError:
        metrics["roc_auc_macro_ovr"] = None

    report = classification_report(
        truth,
        predictions,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    per_class = pd.DataFrame(report).T
    matrix = confusion_matrix(truth, predictions, labels=labels)
    return metrics, per_class, matrix


def prediction_table(
    paths: Sequence[str],
    y_true: Sequence[int],
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    probability = np.asarray(probabilities)
    predictions = probability.argmax(axis=1)
    table = pd.DataFrame(
        {
            "path": list(paths),
            "true_index": np.asarray(y_true, dtype=int),
            "true_class": [class_names[index] for index in y_true],
            "predicted_index": predictions,
            "predicted_class": [class_names[index] for index in predictions],
            "confidence": probability.max(axis=1),
        }
    )
    for index, label in enumerate(class_names):
        table[f"probability_{label}"] = probability[:, index]
    return table


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    class_names: Sequence[str],
    destination: str | Path,
    title: str,
) -> Path:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names))))
    display = ConfusionMatrixDisplay(matrix, display_labels=list(class_names))
    display.plot(ax=axis, cmap="Blues", colorbar=False, xticks_rotation=45)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path
