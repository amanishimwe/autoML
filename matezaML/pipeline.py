"""Lightweight pipeline utilities for low-resource workflows."""

from __future__ import annotations


class Pipeline:
    """Simple sklearn-like pipeline for transformers and an estimator."""

    def __init__(self, steps):
        if not steps or len(steps) < 2:
            raise ValueError("Pipeline requires at least one transformer and one estimator.")
        self.steps = steps

    @property
    def transformers(self):
        return self.steps[:-1]

    @property
    def estimator(self):
        return self.steps[-1]

    def fit(self, X, y):
        X_current = X
        for _, transformer in self.transformers:
            X_current = transformer.fit_transform(X_current, y)

        _, estimator = self.estimator
        estimator.fit(X_current, y)
        return self

    def predict(self, X):
        X_current = X
        for _, transformer in self.transformers:
            X_current = transformer.transform(X_current)

        _, estimator = self.estimator
        return estimator.predict(X_current)
