r"""Register ``metric_matrix`` and ``metric_representation`` dispatch rules.

Covers :class:`~coordinax.manifolds.EmbeddedManifold` paired with any
intrinsic :class:`~coordinax._src.base.AbstractChart`.

The *induced* (pullback) metric on an embedded submanifold is computed as
$g = J^T G J$ where $J$ is the Jacobian of the composition
``chart → intrinsic → Cartesian ambient`` and $G$ is the ambient metric
evaluated at the embedded point.  $G$ is the identity only when the ambient is
Euclidean; for a Lorentzian ambient it is $\eta$, and dropping it would report
a timelike direction as spacelike.  Routing through the Cartesian ambient makes
every ambient output share the single unit ``cart_unit`` (column *i* of $J$ then
has unit ``cart_unit / chart_unit_i``), which makes each summation term
unit-compatible; $G$ in that chart is dimensionless.

All results are wrapped in a :class:`~coordinax._src.metric.matrix.DenseMetric`
because the induced metric is not guaranteed to be diagonal.

"""

__all__: tuple[str, ...] = ()

import jax
import jax.numpy as jnp
import plum
import unxts.linalg as ul

import quaxed.numpy as qnp
import unxt as u

from .manifold import EmbeddedManifold
from coordinax._src.base import AbstractChart  # type: ignore[type-arg]
from coordinax._src.metric.matrix import AbstractMetricMatrix, DenseMetric
from coordinax.internal import pack_nonuniform_unit
from coordinaxs.api.manifolds import metric_matrix, pt_embed

DMLS = u.unit("")


def _gram_values(g: AbstractMetricMatrix) -> jnp.ndarray:
    """Ambient metric as a plain dense array.

    Cartesian ambient coordinates share a single unit, so the ambient metric in
    that chart is dimensionless and its bare values carry the whole content —
    which is what the caller's ``cart_unit^2 / (chart_unit_i * chart_unit_j)``
    result unit assumes.
    """
    m = g.to_dense().matrix
    return m.value if isinstance(m, ul.QuantityMatrix) else m


# =====================================================================
# metric_representation
# =====================================================================


@plum.dispatch
def metric_representation(
    M: EmbeddedManifold, chart: AbstractChart, /
) -> type[DenseMetric]:
    """Embedded manifold in any intrinsic chart → `DenseMetric`.

    >>> import unxt as u
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> from coordinaxs.api.manifolds import metric_representation

    >>> M = cxm.EmbeddedManifold(
    ...     intrinsic=cxm.S2, ambient=cxm.R3,
    ...     embed_map=cxm.TwoSphereIn3D(radius=u.Q(1.0, "km")),
    ... )
    >>> metric_representation(M, cxc.sph2)
    <class 'coordinax._src.metric.matrix.DenseMetric'>

    """
    del M, chart
    return DenseMetric


# =====================================================================
# metric_matrix
# =====================================================================


@plum.dispatch
def metric_matrix(
    M: EmbeddedManifold, point: dict, chart: AbstractChart, /
) -> DenseMetric:
    r"""Induced metric on an embedded submanifold via Jacobian pullback.

    Computes $g_{ij} = \sum_{kl} J^k_i G_{kl} J^l_j$ where $J$ is the Jacobian
    of the composition ``chart → intrinsic → Cartesian ambient`` (the ``chart →
    intrinsic`` leg is the identity when ``chart`` is the intrinsic chart) and
    $G$ is the ambient metric at the embedded point.  $G$ is the identity for a
    Euclidean ambient, so the familiar $J^T J$ is the special case; for a
    Lorentzian ambient $G = \eta$ and the induced metric of a timelike
    direction is correctly negative.

    Routing through Cartesian ambient coordinates makes every ambient output
    share the single unit ``cart_unit`` (so column *i* of $J$ has unit
    ``cart_unit / chart_unit_i``) and makes $G$ dimensionless; each ``g_{ij}``
    term then has a consistent unit ``cart_unit^2 / (chart_unit_i *
    chart_unit_j)`` and the result carries physically correct units.

    Parameters
    ----------
    M : EmbeddedManifold
        An embedded submanifold; carries ``intrinsic``, ``ambient``, and
        ``embed_map`` fields.
    point : dict
        A coordinate dictionary in the passed ``chart``'s coordinates.
    chart : AbstractChart
        The chart in which ``point`` is expressed and in which the metric is
        returned; mapped into the embedding's intrinsic chart when the two
        differ.

    Returns
    -------
    DenseMetric
        Induced metric matrix at ``point``, backed by a
        :class:`~unxts.linalg.QuantityMatrix` with units
        ``cart_unit^2 / (chart_unit_i * chart_unit_j)``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> from coordinaxs.api.manifolds import metric_matrix
    >>> from coordinax._src.metric.matrix import DenseMetric

    Unit sphere — values should be the identity:

    >>> M = cxm.EmbeddedManifold(
    ...     intrinsic=cxm.S2, ambient=cxm.R3,
    ...     embed_map=cxm.TwoSphereIn3D(radius=1.0),
    ... )
    >>> p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> g = metric_matrix(M, p, cxc.sph2)
    >>> isinstance(g, DenseMetric)
    True
    >>> g.matrix.value
    Array([[1., 0.],
           [0., 1.]], dtype=float64)

    Radius-2 sphere — metric scaled by R²:

    >>> M2 = cxm.EmbeddedManifold(
    ...     intrinsic=cxm.S2, ambient=cxm.R3,
    ...     embed_map=cxm.TwoSphereIn3D(radius=u.Q(2.0, "m")),
    ... )
    >>> g2 = metric_matrix(M2, p, cxc.sph2)
    >>> g2.matrix.value
    Array([[4., 0.],
           [0., 4.]], dtype=float64)
    >>> g2.matrix.unit[0, 0]
    Unit("m2 / rad2")

    The metric is returned in the coordinates of the *passed* chart:

    >>> p_ll = {"lon": u.Angle(0.0, "rad"), "lat": u.Angle(jnp.pi / 3, "rad")}
    >>> metric_matrix(M, p_ll, cxc.lonlat_sph2).matrix.value
    Array([[0.25, 0.  ],
           [0.  , 1.  ]], dtype=float64)

    """
    chart_keys = chart.components
    # Use Cartesian ambient so all outputs share cart_unit; column i of J then
    # has unit cart_unit / chart_unit_i, making each g_ij term unit-consistent.
    cart_chart = M.embed_map.ambient.cartesian
    cart_keys = cart_chart.components

    xat, ufrom = pack_nonuniform_unit(point, chart_keys)
    ufrom_ = tuple(uf if uf is not None else DMLS for uf in ufrom)

    # `pt_embed` is the composition chart → intrinsic → ambient → Cartesian; it
    # also checks `chart` against the manifold's atlas.
    at_cart = pt_embed(point, chart, cart_chart, M)
    uto_ = ul.cdict_units(at_cart, cart_keys)
    uto_ = tuple(ut if ut is not None else DMLS for ut in uto_)

    def _embed_cart(x_arr: jnp.ndarray) -> jnp.ndarray:
        q = {k: u.Q(x_arr[i], ufrom_[i]) for i, k in enumerate(chart_keys)}
        # `M.embed_map`, not `M`: the atlas check already ran above, and this
        # runs under jacfwd/vmap.
        q_cart = pt_embed(q, chart, cart_chart, M.embed_map)
        vals = [
            u.ustrip(uto_[j], q_cart[k])  # ty: ignore[not-subscriptable]
            if isinstance(q_cart[k], u.AbstractQuantity)  # ty: ignore[not-subscriptable]
            else qnp.asarray(q_cart[k])  # ty: ignore[not-subscriptable]
            for j, k in enumerate(cart_keys)
        ]
        return qnp.stack(vals)

    def _ambient_gram(y_arr: jnp.ndarray) -> jnp.ndarray:
        """Ambient metric G at the embedded point, as (n_cart, n_cart)."""
        q = {k: u.Q(y_arr[j], uto_[j]) for j, k in enumerate(cart_keys)}
        return _gram_values(metric_matrix(M.ambient, q, cart_chart))

    def _single_metric(x_vec: jnp.ndarray) -> jnp.ndarray:
        j = jax.jacfwd(_embed_cart)(x_vec)  # (n_cart, n_chart)
        # G, not the identity: a Lorentzian ambient contributes the sign.
        return j.T @ _ambient_gram(_embed_cart(x_vec)) @ j  # (n_chart, n_chart)

    # `xat` is (*batch, n_chart) — components last, batch leading. vmap the
    # per-point Jacobian over the flattened batch; a plain jacfwd of a batched
    # input would give a wrong (cross-batch) Jacobian.
    n = len(chart_keys)
    result_vals = jax.vmap(_single_metric)(xat.reshape(-1, n))
    result_vals = result_vals.reshape(*xat.shape[:-1], n, n)

    # g_{ij} unit = uto_[0]² / (ufrom_[i] × ufrom_[j])
    # Valid because all Cartesian coordinates share the same unit.
    result_unit = ul.UnitsMatrix(
        tuple(
            tuple(uto_[0] ** 2 / (ufrom_[i] * ufrom_[j]) for j in range(n))  # ty: ignore[unsupported-operator]
            for i in range(n)
        )
    )
    return DenseMetric(ul.QuantityMatrix(result_vals, unit=result_unit))
