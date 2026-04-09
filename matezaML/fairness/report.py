"""Fairness reporting helpers."""

from __future__ import annotations

import pandas as pd


def fairness_report(y_pred, sensitive_feature):
    """Return per-group positive prediction rates."""
    frame = pd.DataFrame({"y_pred": y_pred, "group": sensitive_feature})
    report = frame.groupby("group", dropna=False)["y_pred"].agg(["mean", "count"])
    return report.rename(columns={"mean": "positive_rate", "count": "samples"})
