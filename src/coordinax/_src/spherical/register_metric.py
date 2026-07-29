"""Register ``metric_matrix`` and ``metric_representation`` dispatch rules.

Covers :class:`~coordinax.manifolds.HyperSphericalManifold` (the unit
$n$-sphere $S^n$) paired with intrinsic angular charts that derive from
:class:`~coordinax._src.spherical.chart.AbstractSphericalHyperSphere`.

The round metric on $S^n$ is diagonal in standard spherical charts, so all
rules return a :class:`~coordinax._src.metric.matrix.DiagonalMetric`.  The
diagonal entries are computed directly via the ``_sine_product_diagonal``
helper, avoiding a full-matrix allocation.

"""

__all__: tuple[str, ...] = ()

from typing import cast

import jax
import jax.numpy as jnp
import plum

import unxt as u
import unxts.linalg as ul
from unxt.quantity import AllowValue

import coordinaxs.api.charts as cxcapi
from .chart import (
    AbstractSphericalHyperSphere,
    NonCanonicalTwoSphere,
    SphericalTwoSphere,
)
from .manifold import HyperSphericalManifold
from coordinax._src.metric.matrix import (
    DenseMetric,
    DiagonalMetric,
    _sine_product_diagonal,
)
from coordinax.internal import CDict, tree_cast_int_bool_to_float

RAD = u.unit("rad")


@plum.dispatch
def metric_representation(
    M: HyperSphericalManifold, chart: AbstractSphericalHyperSphere, /
) -> type[DiagonalMetric]:
    """Return `DiagonalMetric` for a unit $n$-sphere in a standard angular chart.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> cxm.metric_representation(cxm.S2, cxc.sph2)
    <class 'coordinax._src.metric.matrix.DiagonalMetric'>

    """
    del M, chart
    return DiagonalMetric


@plum.dispatch
def metric_matrix(
    M: HyperSphericalManifold, point: CDict, chart: AbstractSphericalHyperSphere, /
) -> DiagonalMetric:
    r"""Round metric on the unit $n$-sphere in a standard angular chart.

    Computes diagonal entries directly via ``_sine_product_diagonal``:

    $$g_{kk} = \prod_{j=0}^{k-1} \sin^2(\theta_j)$$

    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinaxs.api.manifolds import metric_matrix
    >>> from coordinax._src.metric.matrix import DiagonalMetric

    $S^2$ at the equator $\theta = \pi/2$:

    >>> M = cxm.HyperSphericalManifold(2)
    >>> at = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0.0)}
    >>> g = metric_matrix(M, at, cxc.sph2)
    >>> isinstance(g, DiagonalMetric)
    True
    >>> g.diagonal
    Array([1., 1.], dtype=float64)

    $S^2$ at $\theta = \pi/6$:

    >>> at = {"theta": jnp.array(jnp.pi / 6), "phi": jnp.array(0.0)}
    >>> g = metric_matrix(M, at, cxc.sph2)
    >>> round(float(g.diagonal[1]), 10)  # sin\u00b2(\u03c0/6) \u2248 0.25
    0.25

    """
    components = chart.components
    # All angular components except the last (azimuthal) are polar angles
    theta_keys = components[:-1]
    if theta_keys:
        thetas = jnp.stack(
            [
                u.ustrip(AllowValue, u.uconvert_value(RAD, RAD, point[k]))
                for k in theta_keys
            ]
        )
    else:
        thetas = jnp.array([])
    diag = _sine_product_diagonal(thetas, 1.0)
    return DiagonalMetric(diag)


@plum.dispatch
def metric_representation(
    M: HyperSphericalManifold, chart: NonCanonicalTwoSphere, /
) -> type[DenseMetric]:
    """Non-canonical two-sphere charts return a `DenseMetric`.

    Their round metric is obtained by pullback (see `metric_matrix`); it is
    diagonal for LonLat/Math but genuinely dense for LonCosLat, so all three use
    `DenseMetric`.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> cxm.metric_representation(cxm.S2, cxc.loncoslat_sph2)
    <class 'coordinax._src.metric.matrix.DenseMetric'>

    """
    del M, chart
    return DenseMetric


@plum.dispatch
def metric_matrix(
    M: HyperSphericalManifold, point: CDict, chart: NonCanonicalTwoSphere, /
) -> DenseMetric:
    r"""Round metric on a non-canonical two-sphere chart (pullback).

    The nested sine-product used by the canonical dispatch assumes the
    components are polar angles in nested order; that is false for these charts
    (LonLat, Math swap/relabel the polar and azimuthal angles; LonCosLat is
    non-orthogonal). The metric is instead pulled back from the canonical chart:
    ``g = Jc^T diag(1, sin^2 theta) Jc``, where ``Jc`` is the Jacobian of the
    coordinate map ``chart -> SphericalTwoSphere``.

    >>> import math
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    LonLat at lat=40 deg: ``g = diag(cos^2(lat), 1)``:

    >>> at = {"lon": u.Q(0.6, "rad"), "lat": u.Q(math.radians(40), "rad")}
    >>> g = cxm.metric_matrix(cxm.S2, at, cxc.lonlat_sph2)
    >>> [round(float(g.matrix.value[i, i]), 4) for i in (0, 1)]
    [0.5868, 1.0]

    """
    del M
    keys = chart.components
    x0 = tree_cast_int_bool_to_float(
        jnp.stack([jnp.asarray(u.ustrip(AllowValue, RAD, point[k])) for k in keys])
    )

    def to_canon(x: jnp.ndarray) -> jnp.ndarray:
        p = {k: u.Q(x[i], RAD) for i, k in enumerate(keys)}
        s = cast("CDict", cxcapi.pt_map(p, chart, SphericalTwoSphere()))
        return jnp.stack([u.ustrip(RAD, s["theta"]), u.ustrip(RAD, s["phi"])])

    jc = jax.jacfwd(to_canon)(x0)  # (2, n)
    theta = to_canon(x0)[0]
    g_can = jnp.diag(jnp.stack([jnp.ones_like(theta), jnp.sin(theta) ** 2]))
    n = len(keys)
    dmls = ul.UnitsMatrix.full((n, n), "")  # angles -> angles, so g is dimensionless
    return DenseMetric(ul.QM(jc.T @ g_can @ jc, unit=dmls))
