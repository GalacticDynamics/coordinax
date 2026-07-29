"""Dispatch implementations for :func:`coordinaxs.api.manifolds.scale_factors`."""

__all__: tuple[str, ...] = ()


import jax
import jax.numpy as jnp
import plum

import quaxed.numpy as qnp
import unxt as u
import unxts.linalg as ul

import coordinaxs.api.charts as cxcapi
import coordinaxs.api.manifolds as cxmapi
from ._utils import as_quantity_matrix
from coordinax._src.base import AbstractChart, AbstractMetricField
from coordinax._src.custom_types import CDict, OptUSys
from coordinax._src.embedded.metric import PullbackMetric
from coordinax._src.euclidean.scale_factors import _column_squared_norms
from coordinax._src.metric.matrix import DiagonalMetric

DMLS = u.unit("")


@plum.dispatch
def scale_factors(chart: AbstractChart, /, *, at: CDict, usys: OptUSys = None) -> ul.QM:
    """Manifold-level dispatch: delegate to the attached metric.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> at = {
    ...     "r": u.Q(jnp.array(2.0), "km"),
    ...     "theta": u.Angle(jnp.pi / 2, "rad"),
    ...     "phi": u.Angle(jnp.array(0.0), "rad"),
    ... }
    >>> cxm.scale_factors(cxc.sph3d, at=at)
    QM([1., 4., 4.], '(, km2 / rad2, km2 / rad2)')

    """
    return cxmapi.scale_factors(chart.M.metric, chart, at=at, usys=usys)  # ty: ignore[invalid-return-type]


@plum.dispatch
def scale_factors(
    metric: AbstractMetricField,
    chart: AbstractChart,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> ul.QM:
    """Return the diagonal entries of the metric at ``at`` in ``chart``.

    Uses the ``metric_matrix`` dispatch API to compute the metric, then
    extracts the diagonal entries.

    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> metric = cxm.RoundMetric(2)
    >>> at = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0.0)}
    >>> cxm.scale_factors(metric, cxc.sph2, at=at)
    QM([1., 1.], '(, )')

    """
    mm = cxmapi.metric_matrix(chart.M, at, chart)
    if isinstance(mm, DiagonalMetric):
        diag = mm.diagonal
        if isinstance(diag, ul.QM):
            return diag
        units = ul.UnitsMatrix.full(diag.shape[-1], DMLS)
        return ul.QM(diag, unit=units)
    return as_quantity_matrix(mm.matrix).diag()  # ty: ignore[unresolved-attribute]


@plum.dispatch
def scale_factors(
    metric: PullbackMetric,
    chart: AbstractChart,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> ul.QM:
    """Return scale factors for a pullback (induced) metric via Jacobian pullback.

    Computes the Jacobian of the composed embedding ``intrinsic →
    Cartesian ambient`` to obtain a unit-consistent Jacobian where every
    entry has the same unit (``ambient_cart_unit / intrinsic_unit``).
    The squared column norms then give the scale factors with correct units.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> M = cxm.EmbeddedManifold(
    ...     intrinsic=cxm.S2, ambient=cxm.R3,
    ...     embed_map=cxm.TwoSphereIn3D(radius=u.Q(2.0, "m")),
    ... )
    >>> at = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    >>> cxm.scale_factors(M.metric, cxc.sph2, at=at)
    QM([4., 4.], '(m2 / rad2, m2 / rad2)')

    """
    embed_map = metric.embed_map
    ambient_chart = embed_map.ambient
    intrinsic_keys = embed_map.intrinsic.components

    # Use Cartesian ambient chart for a unit-consistent Jacobian.
    # Every column of J_cart has the same per-column unit (cart_unit / intrinsic_unit),
    # which makes _column_squared_norms well-defined with correct units.
    cart_chart = ambient_chart.cartesian
    cart_keys = cart_chart.components

    _qm: ul.QM = cxcapi.carray(at, intrinsic_keys)  # ty: ignore[invalid-assignment]
    xat, ufrom = _qm.value, _qm.unit.to_tuple()
    ufrom_ = tuple(uf if uf is not None else DMLS for uf in ufrom)

    # Evaluate once to determine Cartesian output units
    at_ambient = embed_map.embed(at, usys=usys)
    at_cart = cxcapi.pt_map(at_ambient, ambient_chart, cart_chart)
    uto_ = ul.cdict_units(at_cart, cart_keys)
    uto_ = tuple(ut if ut is not None else DMLS for ut in uto_)

    # Build the unit matrix: J_cart.unit[k][i] = cart_unit_k / intrinsic_unit_i
    unit_matrix = ul.UnitsMatrix(
        tuple(tuple(tj / fi for fi in ufrom_) for tj in uto_)  # ty: ignore[unsupported-operator]
    )

    def _embed_cart(x_arr: jnp.ndarray) -> jnp.ndarray:
        q = {k: u.Q(x_arr[i], ufrom_[i]) for i, k in enumerate(intrinsic_keys)}
        q_ambient = embed_map.embed(q, usys=usys)
        q_cart = cxcapi.pt_map(q_ambient, ambient_chart, cart_chart)
        vals = [
            u.ustrip(uto_[j], q_cart[k])  # ty: ignore[not-subscriptable]
            if isinstance(q_cart[k], u.AbstractQuantity)  # ty: ignore[not-subscriptable]
            else qnp.asarray(q_cart[k])  # ty: ignore[not-subscriptable]
            for j, k in enumerate(cart_keys)
        ]
        return qnp.stack(vals)

    J_arr = jax.jacfwd(_embed_cart)(xat)  # (n_cart, n_intrinsic)
    J_cart = ul.QM(J_arr, unit=unit_matrix)
    return _column_squared_norms(J_cart)
