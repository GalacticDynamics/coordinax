"""Register `act` implementations for various types."""

__all__: tuple[str, ...] = ()

from jaxtyping import Array, ArrayLike
from typing import Any, Final, cast

import plum

import quaxed.numpy as jnp
import unxt as u
from unxt import AbstractQuantity as AbcQ

import coordinax.api.transforms as cxfmapi
import coordinax.charts as cxc
import coordinax.representations as cxr
from .base import AbstractTransform
from coordinax.internal import QuantityMatrix, pack_nonuniform_unit, pack_uniform_unit
from coordinax.internal.custom_types import CDict

_MSG_CHARTS_MATCH: Final = (
    "inferred chart guess_chart(x)={0.__class__.__name__} "
    "does not match provided chart {1.__class__.__name__}"
)

# ===================================================================
# On Array(like) inputs


@plum.dispatch
def act(op: AbstractTransform, tau: Any, x: ArrayLike, /, **kw: Any) -> Array:
    """Apply an operator to an Array(like) object.

    The Array is interpreted as equivalent to the data for a ``Vector``
    with a Cartesian chart (e.g. `coordinax.charts.Cartesian3D`) and
    `coordinax.representations.PointGeometry` geometry.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.transforms as cxfm

    >>> usys = u.unitsystems.si
    >>> x = jnp.asarray([1, 0, 0])  # [m]

    >>> T = cxfm.Translate.from_([1, 0, 0], "km")
    >>> cxfm.act(T, None, x, usys=usys).round(3)  # needs usys
    Array([1001.,    0.,    0.], dtype=float64)

    >>> R = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> cxfm.act(R, None, x).round(3)  # no usys required
    Array([0., 1., 0.], dtype=float64)

    >>> op = R | T  # rotate then translate
    >>> cxfm.act(op, None, x, usys=usys).round(3)
    Array([1000.,    1.,    0.], dtype=float64)

    """
    x_arr = jnp.asarray(x)  # Ensure array type
    chart = cxc.guess_chart(x_arr)  # extract chart
    # Redispatch (op, tau, x) -> (op, tau, x, chart, rep)
    out = cxfmapi.act(op, tau, x, chart, cxr.point, **kw)
    return cast("Array", out)


@plum.dispatch
def act(
    op: AbstractTransform,
    tau: Any,
    x: ArrayLike,
    chart: cxc.AbstractChart,
    /,
    **kw: Any,
) -> Array:
    """Apply an operator to an Array(like) object.

    The Array is interpreted as equivalent to the data for a ``Vector``
    with a Cartesian chart (e.g. `coordinax.charts.Cartesian3D`) and
    `coordinax.representations.PointGeometry` geometry.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> usys = u.unitsystems.si
    >>> x = jnp.asarray([1, 0, 0])  # [m]

    >>> T = cx.Translate.from_([1, 0, 0], "km")
    >>> cx.act(T, None, x, cxc.cart3d, usys=usys).round(3)  # needs usys
    Array([1001.,    0.,    0.], dtype=float64)

    >>> R = cx.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> cx.act(R, None, x, cx.cart3d).round(3) # no usys required
    Array([0., 1., 0.], dtype=float64)

    >>> op = R | T  # rotate then translate
    >>> cx.act(op, None, x, cx.cart3d, usys=usys).round(3)
    Array([1000.,    1.,    0.], dtype=float64)

    """
    # (op, tau, x, chart) -> (op, tau, x, chart, rep)
    out = cxfmapi.act(op, tau, x, chart, cxr.point, **kw)
    return cast("Array", out)


@plum.dispatch
def act(
    op: AbstractTransform,
    tau: Any,
    x: ArrayLike,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    **kw: Any,
) -> Array:
    """Apply an operator to an Array(like) object.

    The Array is interpreted as equivalent to the data for a ``Vector``
    with a Cartesian chart (e.g. `coordinax.charts.Cartesian3D`) and
    `coordinax.representations.PointGeometry` geometry.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> op = cx.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> q = u.Q([1, 0, 0], "km")
    >>> cx.act(op, None, q, cx.cart3d, cx.point).round(3)
    Q([0., 1., 0.], 'km')

    """
    out = cxfmapi.act(op, tau, x, chart, rep.geom_kind, rep, **kw)
    return cast("Array", out)


# ===================================================================
# On Quantity inputs


# Precedence=-1 so it's easily overridden, like by `Identity`.
@plum.dispatch(precedence=-1)  # ty: ignore[no-matching-overload]
def act(op: AbstractTransform, tau: Any, x: AbcQ, /, **kw: Any) -> AbcQ:
    """Apply an operator to a Quantity.

    The Quantity is interpreted as equivalent to the data for a
    `coordinax.Point` with a Cartesian chart (e.g.
    `coordinax.charts.Cartesian3D`) and
    `coordinax.representations.PointGeometry` geometry.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> op = cx.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> q = u.Q([1, 0, 0], "km")
    >>> cx.act(op, None, q).round(3)
    Q([0., 1., 0.], 'km')

    """
    # Redispatch based on the inferred role & chart
    chart = cxc.guess_chart(x)
    rep = cxr.guess_rep(x)
    out = cxfmapi.act(op, tau, x, chart, rep, **kw)
    return cast("AbcQ", out)


@plum.dispatch
def act(
    op: AbstractTransform,
    tau: Any,
    x: AbcQ,
    chart: cxc.AbstractChart,
    /,
    **kw: Any,
) -> AbcQ:
    """Apply operator, routing through dictionary-based implementation.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.transforms as cxfm

    >>> op = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> q = u.Q([1, 0, 0], "km")

    Directly access this registered method, bypassing more efficient methods.

    >>> func = cxfm.act.invoke(cxfm.Rotate, None, u.Q, cxc.Cart3D)
    >>> func(op, None, q, cxc.cart3d).round(3)
    Q([0., 1., 0.], 'km')

    """
    # Get the Cartesian CDict of the input Quantity
    v = cxc.cdict(x, chart)
    # Apply the operator to the CDict representation
    rep = cxr.guess_rep(v)
    nv = cxfmapi.act(op, tau, v, chart, rep, **kw)
    # Stack back to a Quantity (homogeneous unit since Cartesian)
    nva, unit = pack_uniform_unit(nv, keys=chart.components)  # ty: ignore[no-matching-overload]
    return u.Q(jnp.stack(nva, axis=-1), unit)


@plum.dispatch
def act(
    op: AbstractTransform,
    tau: Any,
    x: AbcQ,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    **kw: Any,
) -> AbcQ:
    """Apply operator, routing through dictionary-based implementation.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxfm
    >>> import coordinax.representations as cxr

    >>> op = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> q = u.Q([1, 0, 0], "km")

    Directly access this registered method, bypassing more efficient methods.

    >>> func = cxfm.act.invoke(cxfm.Rotate, None, u.Q, cxc.Cart3D, cxr.Representation)
    >>> func(op, None, q, cxc.cart3d, cxr.point).round(3)
    Q([0., 1., 0.], 'km')

    """
    # Get the Cartesian CDict of the input Quantity
    v = cxc.cdict(x, chart)
    # Act on the CDict representation
    nv = cxfmapi.act(op, tau, v, chart, rep, **kw)
    # Restack to a Quantity (homogeneous unit since Cartesian)
    v, unit = pack_uniform_unit(nv, keys=chart.components)  # ty: ignore[no-matching-overload]
    return u.Q(v, unit)


# ===================================================================
# On QuantityMatrix inputs
#
# Precedence=2 on all QuantityMatrix dispatches so they are preferred over
# the (SpecificTransform, AbstractQuantity) dispatches in rotate.py,
# translate.py, and composed.py (precedence=0) AND over the Identity
# catch-all (precedence=1). Without this, plum sees e.g.
# (Composed, tau, QM) as ambiguous between (Composed, tau, AbcQ) and
# (AbstractTransform, tau, QM).


@plum.dispatch(precedence=2)  # ty: ignore[no-matching-overload]
def act(
    op: AbstractTransform, tau: Any, x: QuantityMatrix, /, **kw: Any
) -> QuantityMatrix:
    """Apply an operator to a ``QuantityMatrix``.

    The chart is inferred from the matrix size and the representation defaults
    to ``point``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.transforms as cxfm
    >>> from coordinax.internal import QuantityMatrix

    >>> op = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> qm = QuantityMatrix(
    ...     jnp.array([1.0, 0.0, 0.0]),
    ...     unit=(u.unit("km"), u.unit("km"), u.unit("km")),
    ... )
    >>> result = cxfm.act(op, None, qm)
    >>> result.value.round(3)
    Array([0., 1., 0.], dtype=float64)

    """
    chart = cxc.guess_chart(x)
    out = cxfmapi.act(op, tau, x, chart, cxr.point, **kw)
    return cast("QuantityMatrix", out)


@plum.dispatch(precedence=2)  # ty: ignore[no-matching-overload]
def act(
    op: AbstractTransform,
    tau: Any,
    x: QuantityMatrix,
    chart: cxc.AbstractChart,
    /,
    **kw: Any,
) -> QuantityMatrix:
    """Apply an operator to a ``QuantityMatrix`` with explicit chart.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxfm
    >>> from coordinax.internal import QuantityMatrix

    >>> qm = QuantityMatrix(
    ...     jnp.array([1.0, 0.0, 0.0]),
    ...     unit=("km", "km", "km"),
    ... )

    >>> op = cxfm.Translate.from_([1, 0, 0], "km")
    >>> result = cxfm.act(op, None, qm, cxc.cart3d)
    >>> result.value.round(3)
    Array([2., 0., 0.], dtype=float64)

    >>> op = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> result = cxfm.act(op, None, qm, cxc.cart3d)
    >>> result.value.round(3)
    Array([0., 1., 0.], dtype=float64)

    """
    out = cxfmapi.act(op, tau, x, chart, cxr.point, **kw)
    return cast("QuantityMatrix", out)


@plum.dispatch(precedence=2)  # ty: ignore[no-matching-overload]
def act(
    op: AbstractTransform,
    tau: Any,
    x: QuantityMatrix,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    **kw: Any,
) -> QuantityMatrix:
    """Apply an operator to a ``QuantityMatrix`` with explicit chart and rep.

    Routes through the CDict-based implementation, then repacks the result
    into a ``QuantityMatrix``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxfm
    >>> import coordinax.representations as cxr
    >>> from coordinax.internal import QuantityMatrix

    >>> op = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> qm = QuantityMatrix(
    ...     jnp.array([1.0, 0.0, 0.0]),
    ...     unit=("km", "km", "km"),
    ... )
    >>> result = cxfm.act(op, None, qm, cxc.cart3d, cxr.point)
    >>> result.value.round(3)
    Array([0., 1., 0.], dtype=float64)

    """
    # Convert QuantityMatrix → CDict
    v = cxc.cdict(x, chart)
    # Act on the CDict
    nv = cxfmapi.act(op, tau, v, chart, rep, **kw)
    # Repack CDict → QuantityMatrix
    arr, units = pack_nonuniform_unit(nv, keys=chart.components)
    return QuantityMatrix(arr, unit=units)


# ===================================================================
# On CDict inputs


@plum.dispatch
def act(op: AbstractTransform, tau: Any, x: CDict, /, **kw: Any) -> CDict:
    """Apply operator to a CDict representation of a vector.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.transforms as cxfm

    >>> op = cxfm.Rotate([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    >>> data = {"x": u.Q(1, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}
    >>> cxfm.act(op, None, data)
    {'x': Q(0, 'km'), 'y': Q(1, 'km'), 'z': Q(0, 'km')}

    """
    # Infer rep and chart from the CDict
    chart = cxc.guess_chart(x)
    rep = cxr.guess_rep(x)

    # Redispatch to the main implementation
    out = cxfmapi.act(op, tau, x, chart, rep, **kw)
    return cast("CDict", out)


@plum.dispatch
def act(
    op: AbstractTransform, tau: Any, x: CDict, chart: cxc.AbstractChart, /, **kw: Any
) -> CDict:
    """Apply operator to a CDict representation of a vector.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.transforms as cxfm
    >>> import coordinax.charts as cxc

    >>> op = cxfm.Rotate([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    >>> data = {"x": u.Q(1, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}
    >>> cxfm.act(op, None, data, cxc.cart3d)
    {'x': Q(0, 'km'), 'y': Q(1, 'km'), 'z': Q(0, 'km')}

    """
    # Infer rep from the CDict
    rep = cxr.guess_rep(x)

    # Redispatch to the main implementation
    out = cxfmapi.act(op, tau, x, chart, rep, **kw)
    return cast("CDict", out)
