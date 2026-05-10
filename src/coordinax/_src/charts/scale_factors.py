"""Euclidean specializations for `coordinax.manifolds.scale_factors`."""

__all__: tuple[str, ...] = ()


import plum

import quaxed.numpy as jnp
import unxt as u

from .d0 import Cart0D
from .d1 import Cart1D
from .d2 import Cart2D
from .d3 import Cart3D
from .dn import CartND
from coordinax._src.base_charts import AbstractDimensionalFlag
from coordinax._src.custom_types import CDict, OptUSys
from coordinax._src.euclidean import EuclideanMetric
from coordinax.internal import QuantityMatrix, UnitsMatrix

DMLS = u.unit("")


@plum.dispatch
def scale_factors(
    metric: EuclideanMetric,
    chart: Cart0D | Cart1D | Cart2D | Cart3D | CartND,
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
        if isinstance(chart, AbstractDimensionalFlag)
        else len(chart.components)
    )
    return QuantityMatrix(
        jnp.ones((n,)), unit=UnitsMatrix(tuple(u.unit("") for _ in range(n)))
    )
