"""Baseline model utilities."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_baseline_classifier(name="logistic_regression", random_state=42):
    """Return a lightweight baseline classifier by name."""
    registry = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        ),
    }
    if name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}.")
    return registry[name]
