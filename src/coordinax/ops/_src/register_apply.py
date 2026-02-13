"""Register apply_op implementations for various types."""

__all__: tuple[str, ...] = ()

from jaxtyping import Array, ArrayLike
from typing import Any, Final

import plum

import quaxed.numpy as jnp
from unxt import AbstractQuantity as AbcQ

import coordinax.api as cxapi
import coordinax.charts as cxc
import coordinax.roles as cxr
from .base import AbstractOperator
from coordinax.api import CsDict

# ===================================================================
# On Array(like) inputs


@plum.dispatch
def apply_op(op: AbstractOperator, tau: Any, x: ArrayLike, /, **kw: Any) -> Array:
    """Apply an operator to an Array(like) object.

    The Array is interpreted as equivalent to the data for a
    {class}`~coordinax.Vector` with a Cartesian chart (e.g.
    {class}`~coordinax.charts.Cartesian3D`) and {class}`~coordinax.roles.Point`
    role.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.ops as cxop

    >>> Rz = jnp.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    >>> op = cxop.Rotate(Rz)
    >>> q = u.Q([1, 0, 0], "km")
    >>> cxop.apply_op(op, None, q)
    Quantity(Array([0, 1, 0], dtype=int64), unit='km')

    """
    x_arr = jnp.asarray(x)
    chart = cxapi.guess_chart(x_arr)
    return cxapi.apply_op(op, tau, cxr.point, chart, x, **kw)


# ===================================================================
# On Quantity inputs


@plum.dispatch
def apply_op(op: AbstractOperator, tau: Any, x: AbcQ, /, **kw: Any) -> AbcQ:
    """Apply an operator to a Quantity.

    The Quantity is interpreted as equivalent to the data for a
    {class}`~coordinax.Vector` with a Cartesian chart (e.g.
    {class}`~coordinax.charts.Cartesian3D`) and {class}`~coordinax.roles.Point`
    role.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.ops as cxop

    >>> Rz = jnp.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    >>> op = cxop.Rotate(Rz)
    >>> q = u.Q([1, 0, 0], "km")
    >>> cxop.apply_op(op, None, q)
    Quantity(Array([0, 1, 0], dtype=int64), unit='km')

    """
    # Redispatch based on the inferred role & chart
    role = cxapi.guess_role(x)
    chart = cxapi.guess_chart(x)
    return cxapi.apply_op(op, tau, role, chart, x, **kw)


_MSG_NOT_CART: Final = (
    "apply_op({op}, ..., Quantity) requires Cartesian components. "
    "chart {name} is not its cartesian_chart."
)


@plum.dispatch
def apply_op(
    op: AbstractOperator,
    tau: Any,
    role: cxr.AbstractRole,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: AbcQ,
    /,
    **kw: Any,
) -> AbcQ:
    """Apply operator, routing through dictionary-based implementation.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.ops as cxop

    >>> Rz = jnp.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    >>> op = cxop.Rotate(Rz)
    >>> q = u.Q([1, 0, 0], "km")

    Directly access this registered method, bypassing more efficient methods.

    >>> func = cxop.apply_op.invoke(cxop.Rotate, None, cxr.Point, cxc.Cart3D, u.Q)
    >>> func(op, None, cxr.point, cxc.cart3d, q)
    Quantity(Array([0, 1, 0], dtype=int64), unit='km')

    >>> func = cxop.apply_op.invoke(cxop.Rotate, None, cxr.PhysDisp, cxc.Cart3D, u.Q)
    >>> func(op, None, cxr.phys_disp, cxc.cart3d, q)
    Quantity(Array([0, 1, 0], dtype=int64), unit='km')

    """
    # Check that chart is Cartesian
    cart = chart.cartesian
    if chart != cart:
        msg = _MSG_NOT_CART.format(op=type(op).__name__, name=type(chart).__name__)
        raise ValueError(msg)

    v = cxapi.cdict(x, chart)
    nv = cxapi.apply_op(op, tau, role, chart, v, **kw)
    return jnp.stack(nv, axis=-1)  # stack back to a Quantity


# ===================================================================
# On CsDict inputs


@plum.dispatch
def apply_op(op: AbstractOperator, tau: Any, x: CsDict, /, **kw: Any) -> CsDict:
    """Apply operator to a CsDict representation of a vector."""
    # Infer role and chart from the CsDict
    role = cxapi.guess_role(x)
    chart = cxapi.guess_chart(x)

    # Redispatch to the main implementation
    return cxapi.apply_op(op, tau, role, chart, x, **kw)
