"""Categorical encoding helpers."""

from __future__ import annotations

import pandas as pd


class CategoricalEncoder:
    """One-hot encoder wrapper using pandas for portability."""

    def __init__(self, drop_first=False):
        self.drop_first = drop_first
        self.categorical_columns_ = []
        self.output_columns_ = []

    def fit(self, X, y=None):
        frame = pd.DataFrame(X).copy()
        self.categorical_columns_ = frame.select_dtypes(exclude="number").columns.tolist()
        encoded = pd.get_dummies(frame, columns=self.categorical_columns_, drop_first=self.drop_first)
        self.output_columns_ = encoded.columns.tolist()
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        encoded = pd.get_dummies(frame, columns=self.categorical_columns_, drop_first=self.drop_first)
        return encoded.reindex(columns=self.output_columns_, fill_value=0)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
