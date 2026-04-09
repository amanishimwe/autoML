"""Shared typing utilities for matezaML."""

from __future__ import annotations

from typing import Protocol


class Transformer(Protocol):
    """Common protocol for preprocessing components."""

    def fit(self, X, y=None):
        ...

    def transform(self, X):
        ...

    def fit_transform(self, X, y=None):
        ...


class Estimator(Protocol):
    """Common protocol for model estimators."""

    def fit(self, X, y):
        ...

    def predict(self, X):
        ...
