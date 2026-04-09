"""Reporting helpers for evaluation outputs."""

from __future__ import annotations

import pandas as pd


def metrics_to_frame(metrics: dict) -> pd.DataFrame:
    """Convert metric dictionary to a tabular report."""
    rows = []
    for key, value in metrics.items():
        if key == "confusion_matrix":
            continue
        rows.append({"metric": key, "value": value})
    return pd.DataFrame(rows)
