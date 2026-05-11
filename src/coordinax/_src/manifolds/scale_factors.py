"""Dispatch implementations for :func:`coordinax.api.manifolds.scale_factors`."""

__all__: tuple[str, ...] = ()

from jaxtyping import Array

import numpy as np
import plum

import unxt as u

import coordinax.api.manifolds as cxmapi
from coordinax._src.base import AbstractChart, AbstractMetric
from coordinax._src.custom_types import CDict, OptUSys
from coordinax.internal import QuantityMatrix, UnitsMatrix


@plum.dispatch
def scale_factors(
    chart: AbstractChart, /, *, at: CDict, usys: OptUSys = None
) -> QuantityMatrix:
    """Manifold-level dispatch: delegate to the attached metric.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> at = {
    ...     "r": u.Q(jnp.array(2.0), "km"),
    ...     "theta": u.Angle(jnp.pi / 2, "rad"),
    ...     "phi": u.Angle(jnp.array(0.0), "rad"),
    ... }
    >>> cxm.scale_factors(cxc.sph3d, at=at)
    QuantityMatrix([1., 4., 4.], '(, km2 / rad2, km2 / rad2)')

    """
    return cxmapi.scale_factors(chart.M.metric, chart, at=at, usys=usys)  # ty: ignore[invalid-return-type]


@plum.dispatch
def scale_factors(
    metric: AbstractMetric,
    chart: AbstractChart,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> QuantityMatrix:
    """Return the diagonal entries of ``metric.metric_matrix(...)`` as a vector.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> metric = cxm.HyperSphericalMetric(2)
    >>> at = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0.0)}
    >>> cxm.scale_factors(metric, cxc.sph2, at=at)
    QuantityMatrix([1., 1.], '(, )')

    """
    return _as_quantity_matrix(metric.metric_matrix(chart, at=at, usys=usys)).diag()


def _as_quantity_matrix(x: QuantityMatrix | Array) -> QuantityMatrix:
    """Convert a numeric matrix into a dimensionless QuantityMatrix."""
    if isinstance(x, QuantityMatrix):
        return x

    n_rows, n_cols = x.shape[-2:]
    units = UnitsMatrix(np.full((n_rows, n_cols), u.unit("")))
    return QuantityMatrix(value=x, unit=units)
