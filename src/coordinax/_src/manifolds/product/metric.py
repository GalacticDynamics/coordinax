"""Product manifold metrics."""

__all__ = ("CartesianProductMetric",)

import dataclasses

from typing import final

import jax
import jax.numpy as jnp

import unxt as u

import coordinax.charts as cxc
from coordinax._src.base_metric import AbstractMetric
from coordinax._src.manifolds.custom_types import CDict, OptUSys
from coordinax.internal import QuantityMatrix, UnitsMatrix


@jax.tree_util.register_static
@final
@dataclasses.dataclass(frozen=True, slots=True)
class CartesianProductMetric(AbstractMetric):
    r"""Canonical product metric on a Cartesian product manifold.

    For factor manifolds $(M_i, g_i)$, the product metric on
    $M = M_1 \times \cdots \times M_k$ is

    $$
    g_{(p_1,\ldots,p_k)}\big((v_1,\ldots,v_k),(w_1,\ldots,w_k)\big)
    = \sum_{i=1}^k g_i(v_i, w_i),
    $$

    which is block diagonal in a product chart.
    """

    factors: tuple[AbstractMetric, ...]
    """Metrics for each factor manifold, in product order."""

    def __post_init__(self) -> None:
        if len(self.factors) == 0:
            raise ValueError("CartesianProductMetric requires at least one factor.")

    @property
    def signature(self) -> tuple[int, ...]:
        """Concatenated factor signatures in product order."""
        return tuple(s for m in self.factors for s in m.signature)

    def metric_matrix(
        self,
        chart: cxc.AbstractChart,
        /,
        *,
        at: CDict,
        usys: OptUSys = None,
    ) -> QuantityMatrix:
        """Return block-diagonal matrix from factor metrics in product chart."""
        if not isinstance(chart, cxc.AbstractCartesianProductChart):
            msg = f"CartesianProductMetric requires a product chart, got {chart!r}."
            raise TypeError(msg)

        if len(chart.factors) != len(self.factors):
            msg = (
                "Product chart factor count does not match "
                "product metric factor count: "
                f"{len(chart.factors)} != {len(self.factors)}"
            )
            raise ValueError(msg)

        parts = chart.split_components(at)
        blocks = tuple(
            _as_quantity_matrix(metric.metric_matrix(c, at=p, usys=usys))
            for metric, c, p in zip(self.factors, chart.factors, parts, strict=True)
        )

        n = sum(block.shape[0] for block in blocks)
        dtype = jnp.promote_types(*(block.value.dtype for block in blocks))
        value = jnp.zeros((n, n), dtype=dtype)
        units = [[u.unit("") for _ in range(n)] for _ in range(n)]

        offset = 0
        for block in blocks:
            block_n = block.shape[0]
            value = value.at[offset : offset + block_n, offset : offset + block_n].set(
                block.value
            )
            for i in range(block_n):
                for j in range(block_n):
                    units[offset + i][offset + j] = block.unit[i, j]
            offset += block_n

        unit_tup = tuple(tuple(row) for row in units)
        return QuantityMatrix(value=value, unit=UnitsMatrix(unit_tup))


def _as_quantity_matrix(x: QuantityMatrix | jax.Array) -> QuantityMatrix:
    """Convert a numeric matrix into a dimensionless QuantityMatrix."""
    if isinstance(x, QuantityMatrix):
        return x

    n = x.shape[0]
    unit_tup = tuple(tuple(u.unit("") for _ in range(n)) for _ in range(n))
    return QuantityMatrix(value=x, unit=UnitsMatrix(unit_tup))
