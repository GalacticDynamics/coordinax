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

import jax.numpy as jnp
import plum

import unxt as u
from unxt.quantity import AllowValue

from .chart import AbstractSphericalHyperSphere
from .manifold import HyperSphericalManifold
from coordinax._src.metric.matrix import DiagonalMetric, _sine_product_diagonal
from coordinax.internal import CDict

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
    >>> from coordinax.api.manifolds import metric_matrix
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
