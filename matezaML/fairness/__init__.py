"""Fairness checks and reports."""

from .demographic_parity import demographic_parity_difference
from .report import fairness_report

__all__ = ["demographic_parity_difference", "fairness_report"]
