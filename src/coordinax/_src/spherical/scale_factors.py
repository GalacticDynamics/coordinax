"""Spherical specializations for `coordinax.manifolds.scale_factors`."""

__all__: tuple[str, ...] = ()

import plum

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

from .metric import RoundMetric
from coordinax._src.base import AbstractChart
from coordinax._src.custom_types import CDict, OptUSys
from coordinax.internal import QMatrix, UnitsMatrix


@plum.dispatch
def scale_factors(
    metric: RoundMetric, chart: AbstractChart, /, *, at: CDict, usys: OptUSys = None
) -> QMatrix:
    r"""Return round-metric diagonal directly without forming the nxn matrix.

    Computes the cumulative-sine diagonal $g_{kk} = \prod_{j<k} \sin^2\theta_j$
    as a 1-D vector, avoiding the O(n^2) cost of :meth:`~.RoundMetric.metric_matrix`.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    Bare angles (no units) → dimensionless QMatrix:

    >>> metric = cxm.RoundMetric(2)
    >>> at = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0.0)}
    >>> cxm.scale_factors(metric, cxc.sph2, at=at)
    QMatrix([1., 1.], '(, )')

    Quantity angles → dimensionless QMatrix:

    >>> at = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> cxm.scale_factors(metric, cxc.sph2, at=at)
    QMatrix([1., 1.], '(, )')

    """
    del metric
    components = chart.components
    ang_unit = usys["angle"] if usys is not None else u.unit("rad")
    angles = jnp.stack(
        [
            u.ustrip(AllowValue, u.uconvert_value(u.unit("rad"), ang_unit, at[k]))
            for k in components[:-1]
        ]
    )
    sin2 = jnp.sin(angles) ** 2
    value = jnp.concatenate([jnp.ones(1, dtype=sin2.dtype), jnp.cumprod(sin2)])
    n = len(components)
    units = UnitsMatrix(tuple(u.unit("") for _ in range(n)))
    return QMatrix(value, unit=units)
