"""Outlier handling utilities."""

from __future__ import annotations

import pandas as pd


class OutlierCapper:
    """Caps numeric outliers using the IQR rule."""

    def __init__(self, iqr_multiplier=1.5):
        self.iqr_multiplier = iqr_multiplier
        self.bounds_ = {}

    def fit(self, X, y=None):
        frame = pd.DataFrame(X).copy()
        for column in frame.select_dtypes(include="number").columns:
            q1 = frame[column].quantile(0.25)
            q3 = frame[column].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            self.bounds_[column] = (lower, upper)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        for column, (lower, upper) in self.bounds_.items():
            frame[column] = frame[column].clip(lower=lower, upper=upper)
        return frame

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
