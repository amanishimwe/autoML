"""matezaML public package API."""

from ._version import __version__
from .pipeline import Pipeline

from . import evaluation, fairness, models, preprocessing

__all__ = [
    "__version__",
    "Pipeline",
    "preprocessing",
    "models",
    "evaluation",
    "fairness",
]
