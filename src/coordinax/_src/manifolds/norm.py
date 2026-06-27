"""Dispatch implementations for `coordinax.api.manifolds.norm`."""

__all__: tuple[str, ...] = ()

from jaxtyping import Array
from typing import Any

import plum

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import is_any_quantity

import coordinax.api.charts as cxcapi
import coordinax.api.manifolds as cxmapi
from coordinax._src.base import AbstractChart, AbstractMetricField
from coordinax._src.charts import Cart0D, Cart1D, Cart2D, Cart3D, CartND
from coordinax._src.custom_types import CDict, OptUSys
from coordinax._src.euclidean import FlatMetric
from coordinax._src.internal import (
    QMatrix,
    pack_nonuniform_unit,
    pack_uniform_unit,
)

# ===================================================================
# Metric Matrix


@plum.dispatch
def norm(G: Array, v: Array, /) -> Array:
    r"""Compute the norm of a vector using a general (possibly curved) metric.

    This assumes ``G`` is evaluated at the correct chart and position, and
    ``v`` components match the chart's ordering.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from coordinax.manifolds import norm

    Identity metric (Euclidean), 3-4-0 vector:

    >>> G = jnp.eye(3)
    >>> v = jnp.array([3, 4, 0])
    >>> norm(G, v)
    Array(5., dtype=float64)

    """
    return jnp.sqrt(jnp.einsum("...i,...ij,...j->...", v, G, v))


array_norm = norm.invoke(Array, Array)  # ty: ignore[unresolved-attribute]


@plum.dispatch
def norm(G: Array, v: u.AbstractQuantity, /) -> u.AbstractQuantity:
    r"""Compute the norm of a vector using a general (possibly curved) metric.

    This assumes ``G`` is evaluated at the correct chart and position, and
    ``v`` components match the chart's ordering.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> from coordinax.manifolds import norm

    Identity metric (Euclidean), quantity vector:

    >>> G = jnp.eye(3)
    >>> v = u.Q(jnp.array([3.0, 4.0, 0.0]), "m/s")
    >>> norm(G, v)
    Q(5., 'm / s')

    """
    unit = u.unit_of(v)
    v_arr = u.ustrip(unit, v)
    out = jnp.sqrt(jnp.einsum("...i,...ij,...j->...", v_arr, G, v_arr))
    return u.Q(out, unit)


@plum.dispatch
def norm(G: QMatrix, v: u.AbstractQuantity, /) -> u.AbstractQuantity:
    r"""Compute the norm of a vector using a general (possibly curved) metric.

    This assumes ``G`` is evaluated at the correct chart and position, and
    ``v`` components match the chart's ordering.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> from coordinax.manifolds import norm

    Identity metric (Euclidean), quantity vector:

    >>> G = jnp.eye(3)
    >>> v = u.Q(jnp.array([3.0, 4.0, 0.0]), "m/s")
    >>> norm(G, v)
    Q(5., 'm / s')

    """
    return jnp.sqrt(jnp.dot(v, jnp.matmul(G, v)))


# TODO: sort out why jaxtyping double registers to restore "Real[Array, "n n"]""
@plum.dispatch
def norm(G: Array, v: CDict, /) -> Array | u.AbstractQuantity:
    r"""Compute the norm of a vector using a general (possibly curved) metric.

    This assumes ``G`` is evaluated at the correct chart and position, and
    ``v`` components match the chart's ordering.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> from coordinax.manifolds import norm

    Identity metric (Euclidean), CDict of quantities:

    >>> G = jnp.eye(3)
    >>> v = {"x": u.Q(3.0, "m/s"), "y": u.Q(4.0, "m/s"), "z": u.Q(0.0, "m/s")}
    >>> norm(G, v)
    Q(5., 'm / s')

    """
    # Pack CDict of quantities into a QMatrix, preserving per-component
    # units.  Then compute v^T G v via QMatrix ops, which handle all unit
    # conversions correctly (including mixed-unit components like m/s and 1/s).
    v_vec, units = pack_nonuniform_unit(v, tuple(v.keys()))
    v_qm = QMatrix(v_vec, unit=units)
    return jnp.sqrt(jnp.dot(v_qm, jnp.matmul(G, v_qm)))

    # # Pack vector components into a uniform array. This also checks that all
    # # components have the same units.
    # v_arr, unit = pack_uniform_unit(v, tuple(v.keys()))

    # # Compute g_ij v^i v^j
    # # For a vector v, the squared norm is v^T @ g @ v
    # # Note: v_arr has shape (..., n) with components in the last dimension
    # norm2 = jnp.einsum("...i,...ij,...j->...", v_arr, G, v_arr)

    # # Return the norm, taking care of units if present
    # return jnp.sqrt(norm2) if unit is None else u.Q(jnp.sqrt(norm2), unit)


# ===========================================================================
# Metric-level


@plum.dispatch
def norm(
    v: CDict,
    metric: AbstractMetricField,
    chart: AbstractChart,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> Any:
    r"""Norm of a vector w.r.t. the metric and chart.

    * **Quantity values** (any unit combination): ``usys`` is optional.
      The result is an ``AbstractQuantity`` whose unit reflects the physical
      dimensions of the norm (e.g. ``m / s`` when components mix ``m/s``
      and ``1/s``).  Mixed-unit components are handled correctly via
      {class}`~coordinax.internal.QMatrix` arithmetic.
    * **Bare jax.Array values**: ``usys`` is **required** (raises
      ``TypeError`` if omitted); a plain ``jax.Array`` is returned.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> metric = cxm.FlatMetric(3)
    >>> at = {"x": jnp.array(0.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}

    CDict of same-unit quantities — returns ``Quantity``:

    >>> v = {"x": u.Q(3.0, "m/s"), "y": u.Q(4.0, "m/s"), "z": u.Q(0.0, "m/s")}
    >>> cxm.norm(v, metric, cxc.cart3d, at=at)
    Q(5., 'm / s')

    CDict of mixed-unit quantities on a spherical chart — also returns
    ``Quantity``.  At ``r = 5 m``, ``θ = π/2`` the metric is
    ``G = diag(1, 25 m², 25 m²)``, so radial velocity ``(1 m/s, 0, 0)``
    has norm ``1 m/s``:

    >>> at_sph = {"r": u.Q(5.0, "m"), "theta": u.Q(jnp.pi / 2, "rad"),
    ...           "phi": u.Q(0.0, "rad")}
    >>> v_mixed = {"r": u.Q(1.0, "m/s"), "theta": u.Q(0.0, "rad/s"),
    ...            "phi": u.Q(0.0, "rad/s")}
    >>> cxm.norm(v_mixed, metric, cxc.sph3d, at=at_sph)
    Q(1., 'm / s')

    CDict of bare arrays (usys required) — returns bare ``Array``:

    >>> v_bare = {"x": jnp.array(3.0), "y": jnp.array(4.0), "z": jnp.array(0.0)}
    >>> cxm.norm(v_bare, metric, cxc.cart3d, at=at, usys=u.unitsystems.si)
    Array(5., dtype=float64)

    This was using the Euclidean metric, but it also works on curved metrics.

    >>> metric = cxm.RoundMetric(2)
    >>> at = {"theta": u.Q(jnp.pi / 4, "rad"), "phi": u.Q(1.0, "rad")}
    >>> v = {"theta": u.Q(0.5, "rad/s"), "phi": u.Q(0.5, "rad/s")}
    >>> cxm.norm(v, metric, cxc.sph2, at=at)
    Q(0.61237244, 'rad / s')

    """
    keys = chart.components
    qty_flags = [is_any_quantity(val) for val in v.values()]
    if any(qty_flags) and not all(qty_flags):
        raise TypeError(
            "norm(): mixed CDict with both Quantity and bare Array values is not "
            "supported. All components must be either all Quantity or all bare Array."
        )
    is_qty = all(qty_flags)

    if metric != chart.M.metric:
        raise ValueError("Metric-level dispatch: metric must match chart's metric")

    if not is_qty and usys is None:
        raise TypeError(
            "norm(): `usys` is required when `v` is a CDict of bare arrays "
            "(no unit information). "
            "Example: pass `usys=unxt.unitsystems.si`."
        )

    # Compute metric matrix; metric_matrix handles both bare and quantity at
    G = metric.metric_matrix(chart, at=at, usys=usys)  # ty: ignore[unresolved-attribute]

    if not is_qty:  # Bare arrays — stack on last axis for correct batch broadcasting
        v_vec = jnp.stack([jnp.asarray(v[k]) for k in keys], axis=-1)
        return jnp.sqrt(jnp.einsum("...i,...ij,...j->...", v_vec, G, v_vec))

    # Pack CDict of quantities into a QMatrix, preserving per-component
    # units.  Then compute v^T G v via QMatrix operations, which handle
    # all unit conversions correctly (including mixed-unit components like m/s
    # and 1/s).
    v_vec, units = pack_nonuniform_unit(v, keys)
    v_qm = QMatrix(v_vec, unit=units)
    return jnp.sqrt(jnp.dot(v_qm, jnp.matmul(G, v_qm)))


@plum.dispatch
def norm(
    v: u.AbstractQuantity,
    metric: AbstractMetricField,
    chart: AbstractChart,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> u.AbstractQuantity:
    r"""Dispatch for a ``AbstractQuantity``.

    The quantity is wrapped in a one-element CDict using the chart's first
    (and only) component name, then the CDict overload is invoked.

    This overload is primarily useful for 1-D manifolds (e.g.
    ``EuclideanManifold(1)`` with a ``Cart1D`` chart).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    Norm of a 1-D displacement on the real line:

    >>> M = cxm.FlatMetric(1)
    >>> at = {"x": jnp.array(0.0)}
    >>> cxm.norm(u.Q(jnp.array([5.0]), "m/s"), M, cxc.cart1d, at=at)
    Q(5., 'm / s')

    """
    v_dict = cxcapi.cdict(v, chart)
    return cxmapi.norm(v_dict, metric, chart, at=at, usys=usys)  # ty: ignore[invalid-return-type]


@plum.dispatch
def norm(
    v: Array,
    metric: AbstractMetricField,
    chart: AbstractChart,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> Array:
    r"""Dispatch for a packed 1-D ``jax.Array`` (v first).

    ``v`` is a stacked array whose entries correspond to the chart's
    component ordering (``chart.components``).  Because the array carries
    no unit information, ``usys`` is **required**; passing
    ``usys=None`` (the default) raises ``TypeError``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    3-4-0 triple in Cartesian 3-D:

    >>> M = cxm.FlatMetric(3)
    >>> at = {"x": jnp.array(0.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
    >>> cxm.norm(jnp.array([3.0, 4.0, 0.0]), M, cxc.cart3d,
    ...          at=at, usys=u.unitsystems.si)
    Array(5., dtype=float64)

    Missing ``usys`` raises ``TypeError``:

    >>> try:
    ...     cxm.norm(jnp.array([3.0, 4.0, 0.0]), M, cxc.sph3d, at=at)
    ... except TypeError as e:
    ...     print(e)
    norm(): `usys` is required when `v` is a bare jax.Array ...

    Unit-sphere S²: ``v = (1, 0)`` at the equator has norm 1:

    >>> M = cxm.RoundMetric(2)
    >>> at_eq = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0.0)}
    >>> cxm.norm(jnp.array([1.0, 0.0]), M, cxc.sph2,
    ...          at=at_eq, usys=u.unitsystems.si)
    Array(1., dtype=float64)

    """
    if usys is None:
        raise TypeError(
            "norm(): `usys` is required when `v` is a bare jax.Array "
            "(no unit information). "
            "Example: pass `usys=unxt.unitsystems.si`."
        )
    G = metric.metric_matrix(chart, at=at, usys=usys)  # ty: ignore[unresolved-attribute]
    return cxmapi.norm(G, v)  # ty: ignore[invalid-return-type]


# ===========================================================================
# Chart-level


@plum.dispatch
def norm(
    v: CDict, chart: AbstractChart, /, *, at: CDict | None = None, usys: OptUSys = None
) -> Any:
    r"""Norm of a CDict vector w.r.t. the chart.

    Dispatches to the metric-level overload via the chart's attached metric.
    For flat (Euclidean) charts ``at`` is optional; for curved metrics it is
    required and passed through to ``metric.metric_matrix``.

    """
    return cxmapi.norm(v, chart.M.metric, chart, at=at, usys=usys)  # type: ignore[return-value]


@plum.dispatch
def norm(
    v: Array, chart: AbstractChart, /, *, at: Any = None, usys: Any = None
) -> Array:
    r"""Norm of a packed ``jax.Array`` w.r.t. the chart.

    Delegates to the metric-level ``norm(v, metric, chart, ...)`` overload via
    the chart's attached metric.  For Euclidean Cartesian charts the fast-path
    ``jnp.linalg.norm`` overload is picked automatically.  For curved charts
    ``at`` and ``usys`` are required; missing ``usys`` raises ``TypeError``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    Euclidean fast-path (no ``at`` needed):

    >>> cxm.norm(jnp.array([3.0, 4.0, 0.0]), cxc.cart3d)
    Array(5., dtype=float64)

    Curved chart (``at`` and ``usys`` required):

    >>> at = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0.0)}
    >>> cxm.norm(jnp.array([1.0, 0.0]), cxc.sph2, at=at, usys=u.unitsystems.si)
    Array(1., dtype=float64)

    """
    return cxmapi.norm(v, chart.M.metric, chart, at=at, usys=usys)  # ty: ignore[invalid-return-type]


@plum.dispatch
def norm(
    v: u.AbstractQuantity, chart: AbstractChart, /, *, at: Any = None, usys: Any = None
) -> u.AbstractQuantity:
    r"""Norm of a single ``AbstractQuantity`` w.r.t. the chart.

    Wraps the quantity in a one-element CDict keyed by the chart's first
    component name, then delegates to the metric-level overload.  Primarily
    useful for 1-D charts (``Cart1D``).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> at = {"x": jnp.array(0.0)}
    >>> cxm.norm(u.Q(5.0, "m/s"), cxc.cart1d, at=at)
    Q(5., 'm / s')

    >>> cxm.norm(u.Q(-3.0, "m"), cxc.cart1d, at=at)
    Q(3., 'm')

    """
    return cxmapi.norm(v, chart.M.metric, chart, at=at, usys=usys)  # ty: ignore[invalid-return-type]


# ===========================================================================
# Euclidean Specializations


@plum.dispatch
def norm(
    v: Array,
    metric: FlatMetric,
    chart: Cart0D | Cart1D | Cart2D | Cart3D | CartND,
    /,
    *,
    at: Any = None,
    usys: Any = None,
) -> Array:
    r"""Short-circuit: Euclidean Cartesian chart -> ``jnp.linalg.norm``.

    When both the metric is flat (``FlatMetric``) and the chart is
    Cartesian (``Cart3D``), the metric matrix is the 3x3 identity, so
    ``‖v‖ = √(vᵀ I v) = ‖v‖₂``.  This overload skips the general matrix
    computation and calls ``jnp.linalg.norm`` directly.  ``at`` and ``usys``
    are ignored.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> M = cxm.FlatMetric(3)
    >>> at = {"x": jnp.array(0.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}

    3-4-0 triple gives norm 5:

    >>> cxm.norm(jnp.array([3.0, 4.0, 0.0]), M, cxc.cart3d, at=at)
    Array(5., dtype=float64)

    ``at`` and ``usys`` are unused and technically skippable in this fast path:

    >>> cxm.norm(jnp.array([1.0, 0.0, 0.0]), M, cxc.cart3d)
    Array(1., dtype=float64)

    """
    del metric, chart, at, usys  # Unused
    return jnp.linalg.norm(v, axis=-1)


@plum.dispatch
def norm(
    data: CDict,
    metric: FlatMetric,
    chart: Cart0D | Cart1D | Cart2D | Cart3D,
    /,
    *,
    at: CDict | None = None,
    usys: Any = None,
) -> u.AbstractQuantity | Array:
    r"""Compute the Euclidean norm of a vector.

    Examples
    --------
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> import unxt as u

    >>> M = cxm.FlatMetric(3)
    >>> v = {"x": u.Q(3, "m"), "y": u.Q(4, "m"), "z": u.Q(0, "m")}
    >>> cxm.norm(v, M, cxc.cart3d)
    Q(5., 'm')

    Works in any dimension:

    >>> M = cxm.FlatMetric(2)
    >>> v2 = {"x": u.Q(3, "km"), "y": u.Q(4, "km")}
    >>> cxm.norm(v2, M, cxc.cart2d)
    Q(5., 'km')

    """
    del metric, at, usys
    v, unit = pack_uniform_unit(data, chart.components)
    vnorm = jnp.linalg.norm(v, axis=-1)
    return u.Q(vnorm, unit) if unit is not None else vnorm
