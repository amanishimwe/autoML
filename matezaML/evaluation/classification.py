"""Evaluation helpers for classification tasks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.utils.multiclass import type_of_target


def classification_metrics(y_true, y_pred, y_proba=None):
    """Compute robust classification metrics for imbalanced data."""
    target_type = type_of_target(y_true)
    is_multiclass = target_type == "multiclass"

    metrics = {
        "f1": f1_score(y_true, y_pred, average="macro" if is_multiclass else "binary", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    if y_proba is not None:
        y_proba_array = np.asarray(y_proba)

        if is_multiclass:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba_array, multi_class="ovr", average="macro")
            metrics["pr_auc"] = average_precision_score(y_true, y_proba_array, average="macro")
        else:
            if y_proba_array.ndim > 1:
                y_score = y_proba_array[:, -1]
            else:
                y_score = y_proba_array
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
            metrics["pr_auc"] = average_precision_score(y_true, y_score)

    return metrics
