from __future__ import annotations

import pandas as pd

from matezaML.pipeline import Pipeline


class FitThenTransformOnly:
    def fit(self, X, y=None):
        self.columns_ = list(pd.DataFrame(X).columns)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        frame["added"] = 1
        return frame


class DummyEstimator:
    def fit(self, X, y):
        self.n_rows_ = len(X)
        return self

    def predict(self, X):
        return [0] * len(X)


def test_pipeline_fit_supports_transformers_without_fit_transform():
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = [0, 1, 0]

    pipe = Pipeline(
        steps=[
            ("t", FitThenTransformOnly()),
            ("m", DummyEstimator()),
        ]
    )

    fitted = pipe.fit(X, y)
    preds = fitted.predict(X)

    assert len(preds) == 3
