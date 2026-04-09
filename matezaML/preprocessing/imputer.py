"""Missing value strategies for tabular data."""

from __future__ import annotations

import pandas as pd


class MissingValueImputer:
    """Simple imputer for numeric and categorical columns."""

    def __init__(self, numeric_strategy="median", categorical_strategy="most_frequent"):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.fill_values_ = {}

    def fit(self, X, y=None):
        frame = pd.DataFrame(X).copy()

        for column in frame.columns:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series):
                if self.numeric_strategy == "mean":
                    self.fill_values_[column] = series.mean()
                else:
                    self.fill_values_[column] = series.median()
            else:
                mode = series.mode(dropna=True)
                self.fill_values_[column] = mode.iloc[0] if not mode.empty else "missing"
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        return frame.fillna(self.fill_values_)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
