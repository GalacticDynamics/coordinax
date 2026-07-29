r"""Register ``metric_matrix`` and ``metric_representation`` dispatch rules.

Covers :class:`~coordinax.manifolds.HyperSphericalManifold` (the unit
$n$-sphere $S^n$) paired with intrinsic angular charts that derive from
:class:`~coordinax._src.spherical.chart.AbstractSphericalHyperSphere`.

The round metric is diagonal in **orthogonal** charts and dense otherwise, so
the rules split two ways:

- Orthogonal charts return a
  :class:`~coordinax._src.metric.matrix.DiagonalMetric`.  In *canonical polar
  order* -- components $(\theta_1, \ldots, \theta_{n-1}, \phi)$ with the
  $\theta_j$ colatitudes -- the diagonal follows the cumulative-sine rule
  $g_{kk} = \prod_{j<k}\sin^2\theta_j$ (via ``_sine_product_diagonal``).
  ``lonlat_sph2`` and ``math_sph2`` merely reorder/rename the angles, so they
  are still diagonal but with their own closed forms.
- ``loncoslat_sph2`` is genuinely non-orthogonal; its metric carries
  off-diagonal terms and is pulled back from the canonical chart,
  $g = J^\top G J$, as a :class:`~coordinax._src.metric.matrix.DenseMetric`.

All rules are batch-safe: coordinate components may carry leading batch axes.
"""

__all__: tuple[str, ...] = ()

from jaxtyping import Array
from typing import Any, cast

import jax
import jax.numpy as jnp
import plum
import unxts.linalg as ul

import unxt as u
from unxt.quantity import AllowValue

import coordinaxs.api.charts as cxcapi
from .chart import (
    AbstractSphericalHyperSphere,
    LonCosLatSphericalTwoSphere,
    LonLatSphericalTwoSphere,
    MathSphericalTwoSphere,
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


def _radians(q: Any, /) -> Array:
    """Strip an angular coordinate to radians, accepting bare arrays."""
    return cast("Array", u.ustrip(AllowValue, u.uconvert_value(RAD, RAD, q)))


# ===================================================================
# metric_representation


@plum.dispatch
def metric_representation(
    M: HyperSphericalManifold, chart: AbstractSphericalHyperSphere, /
) -> type[DiagonalMetric]:
    """Return `DiagonalMetric` for a unit $n$-sphere in an orthogonal chart.

    Covers the canonical hyperspherical charts and the orthogonal two-sphere
    charts ``lonlat_sph2`` and ``math_sph2`` (``loncoslat_sph2`` is handled by
    the more specific `DenseMetric` rule below).

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> cxm.metric_representation(cxm.S2, cxc.sph2)
    <class 'coordinax._src.metric.matrix.DiagonalMetric'>

    >>> cxm.metric_representation(cxm.S2, cxc.lonlat_sph2)
    <class 'coordinax._src.metric.matrix.DiagonalMetric'>

    """
    del M, chart
    return DiagonalMetric


@plum.dispatch
def metric_representation(
    M: HyperSphericalManifold, chart: LonCosLatSphericalTwoSphere, /
) -> type[DenseMetric]:
    """Return `DenseMetric` for the non-orthogonal ``loncoslat_sph2`` chart.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> cxm.metric_representation(cxm.S2, cxc.loncoslat_sph2)
    <class 'coordinax._src.metric.matrix.DenseMetric'>

    """
    del M, chart
    return DenseMetric


# ===================================================================
# metric_matrix


@plum.dispatch
def metric_matrix(
    M: HyperSphericalManifold, point: CDict, chart: AbstractSphericalHyperSphere, /
) -> DiagonalMetric:
    r"""Round metric on the unit $n$-sphere in a canonical polar-order chart.

    Computes diagonal entries directly via ``_sine_product_diagonal``:

    $$g_{kk} = \prod_{j=0}^{k-1} \sin^2(\theta_j)$$

    This holds only when ``chart.components`` are the colatitudes
    $\theta_1, \ldots, \theta_{n-1}$ followed by the azimuth $\phi$; charts that
    order or parameterise their angles differently have their own dispatches
    below.

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
    >>> round(float(g.diagonal[1]), 10)  # sin²(π/6) ≈ 0.25
    0.25

    """
    del M
    components = chart.components
    # All angular components except the last (azimuthal) are polar angles.
    theta_keys = components[:-1]
    if theta_keys:
        thetas = jnp.stack([_radians(point[k]) for k in theta_keys], axis=-1)
    else:
        thetas = jnp.zeros((*jnp.shape(_radians(point[components[-1]])), 0))
    diag = _sine_product_diagonal(thetas, 1.0)
    return DiagonalMetric(diag)


@plum.dispatch
def metric_matrix(
    M: HyperSphericalManifold, point: CDict, chart: LonLatSphericalTwoSphere, /
) -> DiagonalMetric:
    r"""Round metric on $S^2$ in ``lonlat_sph2``.

    With $\mathrm{lat} = \pi/2 - \theta$ and $\mathrm{lon} = \phi$, the round
    metric $\mathrm{d}\theta^2 + \sin^2\theta\,\mathrm{d}\phi^2$ becomes
    $g = \mathrm{diag}(\cos^2 \mathrm{lat}, 1)$ over ``(lon, lat)``.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinaxs.api.manifolds import metric_matrix

    >>> at = {"lon": u.Angle(0.6, "rad"), "lat": u.Angle(jnp.pi / 3, "rad")}
    >>> metric_matrix(cxm.S2, at, cxc.lonlat_sph2).diagonal
    Array([0.25, 1.  ], dtype=float64, weak_type=True)

    """
    del M, chart
    lat = _radians(point["lat"])
    return DiagonalMetric(jnp.stack([jnp.cos(lat) ** 2, jnp.ones_like(lat)], axis=-1))


@plum.dispatch
def metric_matrix(
    M: HyperSphericalManifold, point: CDict, chart: MathSphericalTwoSphere, /
) -> DiagonalMetric:
    r"""Round metric on $S^2$ in ``math_sph2``.

    The mathematics convention swaps the angle names: ``theta`` is the azimuth
    and ``phi`` the polar angle, so $g = \mathrm{diag}(\sin^2\phi, 1)$ over
    ``(theta, phi)``.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinaxs.api.manifolds import metric_matrix

    >>> at = {"theta": u.Angle(0.6, "rad"), "phi": u.Angle(0.9, "rad")}
    >>> metric_matrix(cxm.S2, at, cxc.math_sph2).diagonal
    Array([0.61360105, 1.        ], dtype=float64, weak_type=True)

    """
    del M, chart
    phi = _radians(point["phi"])
    return DiagonalMetric(jnp.stack([jnp.sin(phi) ** 2, jnp.ones_like(phi)], axis=-1))


@plum.dispatch
def metric_matrix(
    M: HyperSphericalManifold, point: CDict, chart: LonCosLatSphericalTwoSphere, /
) -> DenseMetric:
    r"""Round metric on $S^2$ in the non-orthogonal ``loncoslat_sph2`` chart.

    Pulls the canonical metric $G = \mathrm{diag}(1, \sin^2\theta)$ back through
    the transition to ``sph2``: $g = J^\top G J$ with
    $J^k_i = \partial(\theta, \phi)^k / \partial(\mathrm{chart})^i$.  Because
    $x = \mathrm{lon}\cos\mathrm{lat}$ is not orthogonal, the result carries
    off-diagonal terms $g = [[1, x\tan y], [x\tan y, 1 + x^2\tan^2 y]]$.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinaxs.api.manifolds import metric_matrix

    >>> at = {"lon_coslat": u.Angle(0.3, "rad"), "lat": u.Angle(0.5, "rad")}
    >>> metric_matrix(cxm.S2, at, cxc.loncoslat_sph2).matrix.value
    Array([[1.        , 0.16389075],
           [0.16389075, 1.02686018]], dtype=float64, weak_type=True)

    """
    canonical = SphericalTwoSphere(M=M)
    keys = chart.components
    canonical_keys = canonical.components
    # Promote integer/bool angles to float so the pullback jacfwd is well-defined.
    x = tree_cast_int_bool_to_float(
        jnp.stack([_radians(point[k]) for k in keys], axis=-1)  # (*batch, n)
    )

    def _to_canonical(x_vec: Array) -> Array:
        q = {k: u.Q(x_vec[i], RAD) for i, k in enumerate(keys)}
        q_canonical = cast("CDict", cxcapi.pt_map(q, chart, canonical))
        return jnp.stack([u.ustrip(RAD, q_canonical[k]) for k in canonical_keys])

    def _single_metric(x_vec: Array) -> Array:
        j = jax.jacfwd(_to_canonical)(x_vec)  # d(theta, phi) / d(chart)
        # G = diag(1, sin^2(theta)), so (J^T G J)_ij = sum_k J_ki G_kk J_kj.
        theta = _to_canonical(x_vec)[0]
        g_diag = jnp.stack([jnp.ones_like(theta), jnp.sin(theta) ** 2])
        return j.T @ (g_diag[:, None] * j)

    # `x` is (*batch, n) — components last. vmap the per-point Jacobian over the
    # flattened batch; a plain jacfwd of a batched input would give a wrong
    # (cross-batch) Jacobian.
    n = len(keys)
    result = jax.vmap(_single_metric)(x.reshape(-1, n)).reshape(*x.shape[:-1], n, n)
    dmls = ul.UnitsMatrix.full((n, n), "")  # angles -> angles, so g is dimensionless
    return DenseMetric(ul.QM(result, unit=dmls))
