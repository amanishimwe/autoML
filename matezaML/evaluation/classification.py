"""Evaluation helpers for classification tasks."""

from __future__ import annotations

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_proba=None):
    """Compute robust classification metrics for imbalanced data."""
    metrics = {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        # Trapezoidal estimate of PR-AUC without adding extra dependencies.
        metrics["pr_auc"] = float(abs((precision[:-1] * (recall[1:] - recall[:-1])).sum()))

    return metrics
