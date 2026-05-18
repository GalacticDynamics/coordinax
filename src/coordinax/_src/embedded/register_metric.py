r"""Register ``metric_matrix`` and ``metric_representation`` dispatch rules.

Covers :class:`~coordinax.manifolds.EmbeddedManifold` paired with any
intrinsic :class:`~coordinax._src.base.AbstractChart`.

The *induced* (pullback) metric on an embedded submanifold is computed as
$g = J^T J$ where $J$ is the Jacobian of the composition
``intrinsic → Cartesian ambient``.  Routing through the Cartesian ambient
ensures every entry of $J$ carries the *same* unit (``cart_unit / intrinsic_unit``),
which makes $J^T J$ unit-compatible across all summation terms.

All results are wrapped in a :class:`~coordinax._src.metric.matrix.DenseMetric`
because the induced metric is not guaranteed to be diagonal.

"""

__all__: tuple[str, ...] = ()

import jax
import jax.numpy as jnp
import plum

import quaxed.numpy as qnp
import unxt as u

import coordinax.api.charts as cxcapi
from .manifold import EmbeddedManifold
from coordinax._src.base import AbstractChart  # type: ignore[type-arg]
from coordinax._src.metric.matrix import DenseMetric
from coordinax.api.manifolds import metric_matrix
from coordinax.internal import (
    QMatrix,
    UnitsMatrix,
    cdict_units,
    pack_nonuniform_unit,
)

DMLS = u.unit("")


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
    >>> from coordinax.api.manifolds import metric_representation

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

    Computes $g_{ij} = \sum_k J^k_i J^k_j$ where $J$ is the Jacobian of the
    composition ``intrinsic → Cartesian ambient``.  Routing through Cartesian
    ambient coordinates ensures all entries of $J$ share the same unit
    (``cart_unit / intrinsic_unit``), so the matrix product $J^T J$ is
    unit-compatible and the result carries physically correct units.

    Parameters
    ----------
    M : EmbeddedManifold
        An embedded submanifold; carries ``intrinsic``, ``ambient``, and
        ``embed_map`` fields.
    point : dict
        A coordinate dictionary in the *intrinsic* chart coordinates.
    chart : AbstractChart
        The intrinsic chart (passed through for API consistency).

    Returns
    -------
    DenseMetric
        Induced metric matrix at ``point``, backed by a
        :class:`~coordinax.internal.QMatrix` with units
        ``cart_unit^2 / (intrinsic_unit_i * intrinsic_unit_j)``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.manifolds as cxm
    >>> import coordinax.charts as cxc
    >>> from coordinax.api.manifolds import metric_matrix
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
           [0., 1.]], dtype=float64, weak_type=True)

    Radius-2 sphere — metric scaled by R²:

    >>> M2 = cxm.EmbeddedManifold(
    ...     intrinsic=cxm.S2, ambient=cxm.R3,
    ...     embed_map=cxm.TwoSphereIn3D(radius=u.Q(2.0, "m")),
    ... )
    >>> g2 = metric_matrix(M2, p, cxc.sph2)
    >>> g2.matrix.value
    Array([[4., 0.],
           [0., 4.]], dtype=float64, weak_type=True)
    >>> g2.matrix.unit[0, 0]
    Unit("m2 / rad2")

    """
    del chart
    embed_map = M.embed_map
    ambient_chart = embed_map.ambient
    intrinsic_keys = embed_map.intrinsic.components

    # Use Cartesian ambient so every J entry has the same unit
    # (cart_unit / intrinsic_unit).
    cart_chart = ambient_chart.cartesian
    cart_keys = cart_chart.components

    xat, ufrom = pack_nonuniform_unit(point, intrinsic_keys)
    ufrom_ = tuple(uf if uf is not None else DMLS for uf in ufrom)

    at_ambient = embed_map.embed(point, usys=None)
    at_cart = cxcapi.pt_map(at_ambient, ambient_chart, cart_chart)
    uto_ = cdict_units(at_cart, cart_keys)
    uto_ = tuple(ut if ut is not None else DMLS for ut in uto_)

    def _embed_cart(x_arr: jnp.ndarray) -> jnp.ndarray:
        q = {k: u.Q(x_arr[i], ufrom_[i]) for i, k in enumerate(intrinsic_keys)}
        q_ambient = embed_map.embed(q, usys=None)
        q_cart = cxcapi.pt_map(q_ambient, ambient_chart, cart_chart)
        vals = [
            u.ustrip(uto_[j], q_cart[k])  # ty: ignore[not-subscriptable]
            if isinstance(q_cart[k], u.AbstractQuantity)  # ty: ignore[not-subscriptable]
            else qnp.asarray(q_cart[k])  # ty: ignore[not-subscriptable]
            for j, k in enumerate(cart_keys)
        ]
        return qnp.stack(vals)

    J_arr = jax.jacfwd(_embed_cart)(xat)  # (n_cart, n_intrinsic)
    result_vals = J_arr.T @ J_arr  # (n_intrinsic, n_intrinsic)

    # g_{ij} unit = uto_[0]² / (ufrom_[i] × ufrom_[j])
    # Valid because all Cartesian coordinates share the same unit.
    n = len(intrinsic_keys)
    result_unit = UnitsMatrix(
        tuple(
            tuple(uto_[0] ** 2 / (ufrom_[i] * ufrom_[j]) for j in range(n))  # ty: ignore[unsupported-operator]
            for i in range(n)
        )
    )
    return DenseMetric(QMatrix(result_vals, unit=result_unit))
