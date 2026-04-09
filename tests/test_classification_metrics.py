from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from matezaML.evaluation import classification_metrics


def test_classification_metrics_multiclass_supports_macro_scores():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    y_proba = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.15, 0.7, 0.15],
            [0.75, 0.2, 0.05],
            [0.2, 0.3, 0.5],
            [0.05, 0.25, 0.7],
        ]
    )

    metrics = classification_metrics(y_true, y_pred, y_proba=y_proba)

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert metrics["f1"] >= 0.0


def test_classification_metrics_uses_average_precision_for_binary_pr_auc():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.8, 0.2, 0.4, 0.9, 0.6])

    metrics = classification_metrics(y_true, y_pred, y_proba=y_proba)

    expected = average_precision_score(y_true, y_proba)
    assert metrics["pr_auc"] == expected
