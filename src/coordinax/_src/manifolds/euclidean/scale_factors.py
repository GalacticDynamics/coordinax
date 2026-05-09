"""Euclidean specializations for :func:`coordinax.api.manifolds.scale_factors`."""

__all__: tuple[str, ...] = ()

from jaxtyping import Array

import plum

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue, is_any_quantity

import coordinax.charts as cxc
from .metric import EuclideanMetric
from coordinax._src.manifolds.custom_types import CDict, OptUSys
from coordinax.internal import QuantityMatrix, UnitsMatrix

DMLS = u.unit("")


@plum.dispatch
def scale_factors(
    metric: EuclideanMetric,
    chart: cxc.Cart0D | cxc.Cart1D | cxc.Cart2D | cxc.Cart3D | cxc.CartND,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> QuantityMatrix:
    """Fast path for Euclidean metrics in Cartesian charts.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> metric = cxm.EuclideanMetric(3)
    >>> at = {
    ...     "x": u.Q(jnp.array(1.0), "m"),
    ...     "y": u.Q(jnp.array(2.0), "m"),
    ...     "z": u.Q(jnp.array(3.0), "m"),
    ... }
    >>> cxm.scale_factors(metric, cxc.cart3d, at=at)
    QuantityMatrix([1., 1., 1.], '(, , )')

    """
    del metric, at, usys
    n = (
        chart.ndim
        if isinstance(chart, cxc.AbstractDimensionalFlag)
        else len(chart.components)
    )
    return QuantityMatrix(
        jnp.ones((n,)), unit=UnitsMatrix(tuple(u.unit("") for _ in range(n)))
    )


@plum.dispatch
def scale_factors(
    metric: EuclideanMetric,
    chart: cxc.AbstractChart,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> QuantityMatrix:
    """Compute only the Euclidean metric diagonal instead of forming ``J.T @ J``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> metric = cxm.EuclideanMetric(3)
    >>> at = {
    ...     "r": u.Q(jnp.array(2.0), "km"),
    ...     "theta": u.Angle(jnp.pi / 2, "rad"),
    ...     "phi": u.Angle(jnp.array(0.0), "rad"),
    ... }
    >>> cxm.scale_factors(metric, cxc.sph3d, at=at)
    QuantityMatrix([1., 4., 4.], '(, km2 / rad2, km2 / rad2)')

    """
    del metric
    cart_chart = chart.cartesian

    if chart == cart_chart:
        n = len(chart.components)
        return QuantityMatrix(
            jnp.ones((n,)), unit=UnitsMatrix(tuple(u.unit("") for _ in range(n)))
        )

    J = cxc.jac_pt_map(at, chart, cart_chart, usys=usys)
    return _column_squared_norms(J)


def _column_squared_norms(J: QuantityMatrix | Array, /) -> QuantityMatrix:
    """Return ``diag(J.T @ J)`` without forming the full Gram matrix."""
    if isinstance(J, QuantityMatrix):
        return _quantity_column_squared_norms(J)
    return _array_column_squared_norms(J)


def _array_column_squared_norms(J: Array, /) -> QuantityMatrix:
    """Return squared column norms for a dimensionless Jacobian array."""
    value = jnp.einsum("...ji,...ji->...i", J, J)
    n = value.shape[-1]
    unit = UnitsMatrix(tuple(DMLS for _ in range(n)))
    return QuantityMatrix(value, unit)


def _quantity_column_squared_norms(J: QuantityMatrix) -> QuantityMatrix:
    """Return squared column norms for a heterogeneous-unit Jacobian."""
    xs = tuple(_colnorm2(J[:, i]) for i in range(J.shape[-1]))
    units = tuple(u.unit_of(x) if is_any_quantity(x) else DMLS for x in xs)
    value = jnp.stack(
        [u.ustrip(AllowValue, unit, x) for x, unit in zip(xs, units, strict=True)],
        axis=-1,
    )
    return QuantityMatrix(value, unit=UnitsMatrix(units))


def _colnorm2(column: QuantityMatrix) -> u.AbstractQuantity | Array:
    """Return the squared norm of a single Jacobian column."""
    return jnp.dot(column, column)
