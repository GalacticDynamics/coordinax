"""Minkowski specializations for `coordinax.manifolds.scale_factors`."""

__all__: tuple[str, ...] = ()

import jax.numpy as jnp
import plum

import unxt as u

from .charts import MinkowskiCT
from .metric import MinkowskiMetric
from coordinax._src.custom_types import CDict, OptUSys
from coordinax.internal import QMatrix, UnitsMatrix


@plum.dispatch
def scale_factors(
    metric: MinkowskiMetric, chart: MinkowskiCT, /, *, at: CDict, usys: OptUSys = None
) -> QMatrix:
    r"""Return the Minkowski metric diagonal $\eta = \operatorname{diag}(-1,1,1,1)$.

    In the canonical `coordinax.charts.MinkowskiCT` chart the metric is
    constant, so ``at`` is ignored and the result does not depend on the base
    point.

    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> metric = cxm.MinkowskiMetric()
    >>> at = {"ct": jnp.array(0.0), "x": jnp.array(1.0),
    ...       "y": jnp.array(0.0), "z": jnp.array(0.0)}
    >>> cxm.scale_factors(metric, cxc.minkowskict, at=at)
    QMatrix([-1.,  1.,  1.,  1.], '(, , , )')

    """
    del chart, at, usys
    n = metric.ndim
    value = jnp.array(list(metric.signature), dtype=float)
    units = UnitsMatrix(tuple(u.unit("") for _ in range(n)))
    return QMatrix(value, unit=units)
