"""Null metric."""

__all__ = ("NoMetric", "no_metric")

import dataclasses

from jaxtyping import Array
from typing import Any, final

import jax

from coordinax._src.base import AbstractMetric
from coordinax.internal import QuantityMatrix, UnitsMatrix


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class NoMetric(AbstractMetric):
    """A degenerate placeholder metric with no geometry.

    ``NoMetric`` is a sentinel value used when a metric object is required
    by the API but none has been specified by the user.

    - ``ndim == -1`` signals "no metric specified".
    - ``metric_matrix(chart, at)`` always raises ``NoGlobalCartesianChartError``.

    """

    @property
    def ndim(self) -> int:
        """Stand-in dimension of the degenerate metric."""
        return False

    @property
    def signature(self) -> tuple[int, ...]:
        """Signature of the degenerate metric."""
        return ()

    def metric_matrix(self, *args: Any, **kwargs: Any) -> QuantityMatrix | Array:
        r"""Compute the metric tensor $g_{ij}$ at base point ``at``."""
        del args, kwargs
        return QuantityMatrix(jax.numpy.array([]), UnitsMatrix(""))


no_metric = NoMetric()
"""Canonical instance of `coordinax.manifolds.NoMetric`."""
