"""Dispatch implementations for :func:`coordinax.api.manifolds.scale_factors`."""

__all__: tuple[str, ...] = ()

from jaxtyping import Array

import jax
import jax.numpy as jnp
import numpy as np
import plum

import quaxed.numpy as qnp
import unxt as u

import coordinax.api.charts as cxcapi
import coordinax.api.manifolds as cxmapi
from coordinax._src.base import AbstractChart, AbstractMetricField
from coordinax._src.custom_types import CDict, OptUSys
from coordinax._src.embedded.metric import PullbackMetric
from coordinax._src.euclidean.scale_factors import _column_squared_norms as _csn
from coordinax._src.metric.matrix import DiagonalMetric
from coordinax.internal import (
    QMatrix,
    UnitsMatrix,
    cdict_units,
    pack_nonuniform_unit,
)

DMLS = u.unit("")


@plum.dispatch
def scale_factors(
    chart: AbstractChart, /, *, at: CDict, usys: OptUSys = None
) -> QMatrix:
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
    QMatrix([1., 4., 4.], '(, km2 / rad2, km2 / rad2)')

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
) -> QMatrix:
    """Return the diagonal entries of the metric at ``at`` in ``chart``.

    Uses the ``metric_matrix`` dispatch API to compute the metric, then
    extracts the diagonal entries.

    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> metric = cxm.RoundMetric(2)
    >>> at = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0.0)}
    >>> cxm.scale_factors(metric, cxc.sph2, at=at)
    QMatrix([1., 1.], '(, )')

    """
    mm = cxmapi.metric_matrix(chart.M, at, chart)
    if isinstance(mm, DiagonalMetric):
        diag = mm.diagonal
        if isinstance(diag, QMatrix):
            return diag
        units = UnitsMatrix(tuple(DMLS for _ in range(diag.shape[-1])))
        return QMatrix(diag, unit=units)
    return _as_quantity_matrix(mm.matrix).diag()  # ty: ignore[unresolved-attribute]


def _as_quantity_matrix(x: QMatrix | Array) -> QMatrix:
    """Convert a numeric matrix into a dimensionless QMatrix."""
    if isinstance(x, QMatrix):
        return x

    n_rows, n_cols = x.shape[-2:]
    units = UnitsMatrix(np.full((n_rows, n_cols), DMLS))
    return QMatrix(value=x, unit=units)


@plum.dispatch
def scale_factors(
    metric: PullbackMetric,
    chart: AbstractChart,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> QMatrix:
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
    QMatrix([4., 4.], '(m2 / rad2, m2 / rad2)')

    """
    embed_map = metric.embed_map
    ambient_chart = embed_map.ambient
    intrinsic_keys = embed_map.intrinsic.components

    # Use Cartesian ambient chart for a unit-consistent Jacobian.
    # Every column of J_cart has the same per-column unit (cart_unit / intrinsic_unit),
    # which makes _column_squared_norms well-defined with correct units.
    cart_chart = ambient_chart.cartesian
    cart_keys = cart_chart.components

    xat, ufrom = pack_nonuniform_unit(at, intrinsic_keys)
    ufrom_ = tuple(uf if uf is not None else DMLS for uf in ufrom)

    # Evaluate once to determine Cartesian output units
    at_ambient = embed_map.embed(at, usys=usys)
    at_cart = cxcapi.pt_map(at_ambient, ambient_chart, cart_chart)
    uto_ = cdict_units(at_cart, cart_keys)
    uto_ = tuple(ut if ut is not None else DMLS for ut in uto_)

    # Build the unit matrix: J_cart.unit[k][i] = cart_unit_k / intrinsic_unit_i
    unit_matrix = UnitsMatrix(
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
    J_cart = QMatrix(J_arr, unit=unit_matrix)
    return _column_squared_norms(J_cart)


def _column_squared_norms(J: QMatrix | Array) -> QMatrix:
    """Return the squared column norms of a Jacobian matrix as a QMatrix."""
    return _csn(J)
