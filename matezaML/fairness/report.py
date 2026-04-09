"""Fairness reporting helpers."""

from __future__ import annotations

import pandas as pd

from .demographic_parity import _positive_indicator


def fairness_report(y_pred, sensitive_feature, positive_label=None):
    """Return per-group positive prediction rates."""
    frame = pd.DataFrame({"group": sensitive_feature})
    frame["positive"] = _positive_indicator(y_pred, positive_label)
    report = frame.groupby("group", dropna=False)["positive"].agg(["mean", "count"])
    return report.rename(columns={"mean": "positive_rate", "count": "samples"})
