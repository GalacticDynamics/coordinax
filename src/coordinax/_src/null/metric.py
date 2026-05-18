"""Null metric."""

__all__ = ("NoMetric", "no_metric")

import dataclasses

from typing import final

import jax

from coordinax._src.base import AbstractMetricField


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class NoMetric(AbstractMetricField):
    """A degenerate placeholder metric with no geometry.

    ``NoMetric`` is a sentinel value used when a metric object is required
    by the API but none has been specified by the user.

    - ``ndim == False`` signals "no metric specified".

    """

    @property
    def ndim(self) -> int:
        """Stand-in dimension of the degenerate metric."""
        return False

    @property
    def signature(self) -> tuple[int, ...]:
        """Signature of the degenerate metric."""
        return ()


no_metric = NoMetric()
"""Canonical instance of `coordinax.manifolds.NoMetric`."""
