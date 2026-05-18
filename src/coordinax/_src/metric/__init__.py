"""Metrics — intrinsic metric fields, matrix representations, and dispatch API."""

__all__ = (
    "AbstractMetricField",
    "AbstractDiagonalMetricField",
    "RoundMetric",
    "AbstractMetricMatrix",
    "DiagonalMetric",
    "DenseMetric",
    "metric_matrix",
    "metric_representation",
)

from .api import metric_matrix, metric_representation
from .field import AbstractDiagonalMetricField, AbstractMetricField, RoundMetric
from .matrix import AbstractMetricMatrix, DenseMetric, DiagonalMetric
