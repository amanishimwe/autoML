"""Feature scaling helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Standard scaling for numeric columns."""

    def __init__(self):
        self.scaler_ = StandardScaler()
        self.numeric_columns_ = []

    def fit(self, X, y=None):
        frame = pd.DataFrame(X).copy()
        self.numeric_columns_ = frame.select_dtypes(include="number").columns.tolist()
        if self.numeric_columns_:
            self.scaler_.fit(frame[self.numeric_columns_])
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        if self.numeric_columns_:
            frame[self.numeric_columns_] = self.scaler_.transform(frame[self.numeric_columns_])
        return frame

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
