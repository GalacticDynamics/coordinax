"""Product manifold metrics."""

__all__ = ("ProductMetric",)

import dataclasses

from typing import final

import jax

from coordinax._src.base import AbstractMetricField


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class ProductMetric(AbstractMetricField):
    r"""Canonical product metric on a Cartesian product manifold.

    For factor manifolds $(M_i, g_i)$, the product metric on
    $M = M_1 \times \cdots \times M_k$ is

    $$
    g_{(p_1,\ldots,p_k)}\big((v_1,\ldots,v_k),(w_1,\ldots,w_k)\big)
    = \sum_{i=1}^k g_i(v_i, w_i),
    $$

    which is block diagonal in a product chart.
    """

    factors: tuple[AbstractMetricField, ...]
    """Metrics for each factor manifold, in product order."""

    def __post_init__(self) -> None:
        if len(self.factors) == 0:
            raise ValueError("ProductMetric requires at least one factor.")

    @property
    def signature(self) -> tuple[int, ...]:
        """Concatenated factor signatures in product order."""
        return tuple(s for m in self.factors for s in m.signature)
