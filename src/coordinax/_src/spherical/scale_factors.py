"""Spherical specializations for `coordinax.manifolds.scale_factors`."""

__all__: tuple[str, ...] = ()

from typing import cast

import plum

import quaxed.numpy as jnp
import unxt as u
import unxts.linalg as ul
from unxt.quantity import AllowValue

import coordinaxs.api.manifolds as cxmapi
from .chart import LonCosLatSphericalTwoSphere, RelabeledTwoSphere
from .manifold import HyperSphericalManifold
from .metric import RoundMetric
from coordinax._src.base import AbstractChart
from coordinax._src.custom_types import OptUSys
from coordinax._src.metric.matrix import DiagonalMetric
from coordinaxs.api.custom_types import CDict


@plum.dispatch
def scale_factors(
    metric: RoundMetric,
    chart: RelabeledTwoSphere,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> ul.QuantityMatrix:
    r"""Scale factors for the orthogonal relabelled two-sphere charts.

    ``LonLat`` and ``Math`` relabel/swap the canonical angles, so the nested
    sine-product does not apply -- but they stay *orthogonal*, so the metric is
    diagonal and its diagonal is the answer:

    - ``LonLat`` ``(lon, lat)``: ``diag(cos^2(lat), 1)``
    - ``Math`` ``(theta, phi)``: ``diag(sin^2(phi), 1)``

    Taken from `metric_matrix` rather than re-derived, so there is one source of
    truth for the pullback.

    >>> import math
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> at = {"lon": u.Q(0.6, "rad"), "lat": u.Q(math.radians(40), "rad")}
    >>> h = cxm.scale_factors(cxm.RoundMetric(2), cxc.lonlat_sph2, at=at)
    >>> [round(float(v), 4) for v in h.value]
    [0.5868, 1.0]

    Bare angle values are interpreted via ``usys["angle"]`` when a unit system
    is supplied:

    >>> usys = u.unitsystem("m", "s", "kg", "deg")
    >>> h = cxm.scale_factors(cxm.RoundMetric(2), cxc.lonlat_sph2,
    ...                       at={"lon": 30.0, "lat": 40.0}, usys=usys)
    >>> [round(float(v), 4) for v in h.value]
    [0.5868, 1.0]

    """
    # `metric_matrix` interprets bare angle values as radians, so normalise
    # `at` through the chart's angle unit first (radians when no `usys`).
    rad = u.unit("rad")
    ang_unit = usys["angle"] if usys is not None else rad
    at_rad = {k: u.uconvert_value(rad, ang_unit, v) for k, v in at.items()}
    # `metric_matrix` already returns a DiagonalMetric for these charts, so the
    # diagonal *is* the answer -- no dense matrix is ever formed.
    g = cast(
        "DiagonalMetric",
        cxmapi.metric_matrix(HyperSphericalManifold(metric.ndim), at_rad, chart),
    )
    return cast("ul.QuantityMatrix", g.diagonal)


@plum.dispatch
def scale_factors(
    metric: RoundMetric,
    chart: LonCosLatSphericalTwoSphere,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> ul.QuantityMatrix:
    """`LonCosLat` is non-orthogonal, so it has no scale factors at all."""
    del metric, at, usys
    msg = (
        "scale_factors is a diagonal (orthogonal-frame) concept and "
        f"{type(chart).__name__} is non-orthogonal: its round metric has a "
        "nonzero off-diagonal (g_01 = lon_coslat * tan(lat)), so no set of "
        "per-axis factors reproduces it. Use coordinax.manifolds.metric_matrix."
    )
    raise NotImplementedError(msg)


@plum.dispatch
def scale_factors(
    metric: RoundMetric, chart: AbstractChart, /, *, at: CDict, usys: OptUSys = None
) -> ul.QuantityMatrix:
    r"""Return round-metric diagonal directly without forming the nxn matrix.

    Computes the cumulative-sine diagonal $g_{kk} = \prod_{j<k} \sin^2\theta_j$
    as a 1-D vector, avoiding the O(n^2) cost of ``RoundMetric.metric_matrix``.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    Bare angles (no units) → dimensionless QuantityMatrix:

    >>> metric = cxm.RoundMetric(2)
    >>> at = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0.0)}
    >>> cxm.scale_factors(metric, cxc.sph2, at=at)
    QM([1., 1.], '(, )')

    Quantity angles → dimensionless QuantityMatrix:

    >>> at = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> cxm.scale_factors(metric, cxc.sph2, at=at)
    QM([1., 1.], '(, )')

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
    units = ul.UnitsMatrix.full(n, "")
    return ul.QM(value, unit=units)
