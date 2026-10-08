r"""Register ``metric_matrix`` and ``metric_representation`` dispatch rules.

Covers :class:`~coordinax.manifolds.EmbeddedManifold` paired with any
intrinsic :class:`~coordinax._src.base.AbstractChart`.

The *induced* (pullback) metric on an embedded submanifold is computed as
$g = J^T G J$ where $J$ is the Jacobian of the composition
``chart → intrinsic → Cartesian ambient`` and $G$ is the ambient metric
evaluated at the embedded point.  $G$ is the identity only when the ambient is
Euclidean; for a Lorentzian ambient it is $\eta$, and dropping it would report
a timelike direction as spacelike.  Every Cartesian ambient output is converted
to a single unit ``cart_unit`` (column *i* of $J$ then has unit
``cart_unit / point_unit_i``), which makes each summation term
unit-compatible; $G$ in that chart is dimensionless.  The embed map's output
decides the units, so outputs of differing dimension are refused.  Separately,
a Quantity output next to a bare one is refused, since the bare one has no unit
to convert from.

All results are wrapped in a :class:`~coordinax._src.metric.matrix.DenseMetric`
because the induced metric is not guaranteed to be diagonal.

"""

__all__: tuple[str, ...] = ()

from typing import cast

import jax
import jax.numpy as jnp
import plum

import quaxed.numpy as qnp
import unxt as u
import unxts.linalg as ul
from unxt.quantity import AllowValue

import coordinaxs.api.charts as cxcapi
from .manifold import EmbeddedManifold
from coordinax._src.base import AbstractChart  # type: ignore[type-arg]
from coordinax._src.metric.matrix import AbstractMetricMatrix, DenseMetric
from coordinaxs.api.manifolds import metric_matrix, pt_embed

DMLS = u.unit("")


def _gram_values(g: AbstractMetricMatrix) -> jnp.ndarray:
    """Ambient metric as a plain dense array.

    Cartesian ambient coordinates share a single unit, so the ambient metric in
    that chart is dimensionless and its bare values carry the whole content —
    which is what the caller's ``cart_unit^2 / (point_unit_i * point_unit_j)``
    result unit assumes. ``AllowValue`` passes a bare matrix through and strips
    a dimensionless one; a unitful one raises rather than losing its unit.
    """
    # `cast`: `ustrip` is typed to return `object`.
    return cast("jnp.ndarray", u.ustrip(AllowValue, "", g.to_dense().matrix))


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

    Every Cartesian ambient output is converted to one unit ``cart_unit`` (the
    first component's), so column *i* of $J$ has unit ``cart_unit /
    point_unit_i`` and $G$ is dimensionless; each ``g_{ij}`` term then has a
    consistent unit ``cart_unit^2 / (point_unit_i * point_unit_j)``, where
    ``point_unit_i`` is the unit of the point's *i*-th coordinate.  Outputs
    of differing dimension (e.g. a length and a time) have no consistent sum
    and raise `ValueError`.  Separately, the outputs must be all Quantity or
    all bare: a Quantity next to a bare one raises `TypeError`, since the bare
    one has no unit to convert from.  All-bare outputs are taken as pure
    numbers, as from ``TwoSphereIn3D(radius=1.0)``; the result's unit then
    comes from the point's coordinates alone.  Bare point coordinates are passed
    to the embed map bare.

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
        ``cart_unit^2 / (point_unit_i * point_unit_j)``.

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
    # has unit cart_unit / point_unit_i, making each g_ij term unit-consistent.
    cart_chart = M.embed_map.ambient.cartesian
    cart_keys = cart_chart.components

    _qm: ul.QM = cxcapi.carray(point, chart_keys)  # ty: ignore[invalid-assignment]
    # `carray` gives a bare component the dimensionless unit, so `ufrom_` has no
    # `None`s. Keep which components were bare: they go back to the embed map
    # bare, as the caller passed them, not as dimensionless Quantities.
    xat, ufrom_ = _qm.value, _qm.unit.to_tuple()
    point_qty = [isinstance(point[k], u.AbstractQuantity) for k in chart_keys]

    # `pt_embed` is the composition chart → intrinsic → ambient → Cartesian; it
    # also checks `chart` against the manifold's atlas.
    at_cart = pt_embed(point, chart, cart_chart, M)
    uto = ul.cdict_units(at_cart, cart_keys)
    # The pullback sums the rows of J, so every Cartesian component must be in
    # one unit. The embed map's output decides the units, not the chart, so
    # enforce it. A bare value next to a Quantity has no unit to put it in;
    # refuse rather than guess, as `quadratic_form` does.
    bare = [ut is None for ut in uto]
    if any(bare) and not all(bare):
        got = ", ".join(
            f"{k}: {'bare' if ut is None else ut}"
            for k, ut in zip(cart_keys, uto, strict=True)
        )
        msg = (
            "metric_matrix(): the embedding mixes Quantity and bare Cartesian "
            f"components ({got}). All must be either Quantity or bare."
        )
        raise TypeError(msg)
    uto_ = tuple(ut if ut is not None else DMLS for ut in uto)
    # All Quantity: rescale same-dimension components (m vs km) to the first's
    # unit, and refuse mixed dimensions, which have no consistent sum.
    cart_unit = uto_[0]
    cart_dim = u.dimension_of(cart_unit)
    if any(u.dimension_of(ut) != cart_dim for ut in uto_[1:]):
        got = ", ".join(
            f"{k}: {str(ut) or 'dimensionless'}"
            for k, ut in zip(cart_keys, uto_, strict=True)
        )
        msg = (
            "metric_matrix(): the induced metric needs every Cartesian ambient "
            f"component in one dimension, but the embedding gives {got}."
        )
        raise ValueError(msg)

    def _embed_cart(x_arr: jnp.ndarray) -> jnp.ndarray:
        q = {
            k: u.Q(x_arr[i], ufrom_[i]) if point_qty[i] else x_arr[i]
            for i, k in enumerate(chart_keys)
        }
        # `M.embed_map`, not `M`: the atlas check already ran above, and this
        # runs under jacfwd/vmap.
        q_cart = pt_embed(q, chart, cart_chart, M.embed_map)
        vals = [
            u.ustrip(cart_unit, q_cart[k])  # ty: ignore[not-subscriptable]
            if isinstance(q_cart[k], u.AbstractQuantity)  # ty: ignore[not-subscriptable]
            else qnp.asarray(q_cart[k])  # ty: ignore[not-subscriptable]
            for k in cart_keys
        ]
        return qnp.stack(vals)

    def _ambient_gram(y_arr: jnp.ndarray) -> jnp.ndarray:
        """Ambient metric G at the embedded point, as (n_cart, n_cart)."""
        q = {k: u.Q(y_arr[j], cart_unit) for j, k in enumerate(cart_keys)}
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

    # g_{ij} unit = cart_unit² / (ufrom_[i] × ufrom_[j]); valid because every
    # Cartesian component was put in `cart_unit` above.
    result_unit = ul.UnitsMatrix(
        tuple(
            tuple(cart_unit**2 / (ufrom_[i] * ufrom_[j]) for j in range(n))  # ty: ignore[unsupported-operator]
            for i in range(n)
        )
    )
    return DenseMetric(ul.QuantityMatrix(result_vals, unit=result_unit))
