from __future__ import annotations

import pandas as pd
import pytest

from matezaML.fairness import demographic_parity_difference, fairness_report


def test_demographic_parity_difference_accepts_numeric_predictions():
    y_pred = [1, 0, 1, 0]
    sensitive_feature = ["A", "A", "B", "B"]

    result = demographic_parity_difference(y_pred, sensitive_feature)

    assert result == 0.0


def test_fairness_functions_require_positive_label_for_non_numeric_predictions():
    y_pred = ["yes", "no", "yes", "no"]
    sensitive_feature = ["A", "A", "B", "B"]

    with pytest.raises(ValueError):
        demographic_parity_difference(y_pred, sensitive_feature)

    with pytest.raises(ValueError):
        fairness_report(y_pred, sensitive_feature)


def test_fairness_report_handles_non_numeric_with_positive_label():
    y_pred = ["yes", "no", "yes", "no"]
    sensitive_feature = ["A", "A", "B", "B"]

    report = fairness_report(y_pred, sensitive_feature, positive_label="yes")

    assert isinstance(report, pd.DataFrame)
    assert set(report.columns) == {"positive_rate", "samples"}
