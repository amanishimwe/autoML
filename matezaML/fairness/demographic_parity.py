"""Fairness metrics for tabular classification outputs."""

from __future__ import annotations

import pandas as pd


def demographic_parity_difference(y_pred, sensitive_feature):
    """Difference between highest and lowest positive prediction rates."""
    frame = pd.DataFrame({"y_pred": y_pred, "group": sensitive_feature})
    group_rates = frame.groupby("group", dropna=False)["y_pred"].mean()
    return float(group_rates.max() - group_rates.min())
