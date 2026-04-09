"""Fairness metrics for tabular classification outputs."""

from __future__ import annotations

import pandas as pd


def _positive_indicator(y_pred, positive_label):
    series = pd.Series(y_pred)

    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    if positive_label is None:
        raise ValueError(
            "Non-numeric predictions require `positive_label` to define the positive class."
        )

    return (series == positive_label).astype(float)


def demographic_parity_difference(y_pred, sensitive_feature, positive_label=None):
    """Difference between highest and lowest positive prediction rates."""
    frame = pd.DataFrame({"group": sensitive_feature})
    frame["positive"] = _positive_indicator(y_pred, positive_label)
    group_rates = frame.groupby("group", dropna=False)["positive"].mean()
    return float(group_rates.max() - group_rates.min())
