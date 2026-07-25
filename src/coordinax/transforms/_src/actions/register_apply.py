"""Register `act` implementations for various types."""

__all__: tuple[str, ...] = ()

from jaxtyping import Array, ArrayLike
from typing import Any, TypeAlias, cast

import plum
import unxts.linalg as ul

import unxt as u
from unxt import AbstractQuantity as AbcQ

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinaxs.api.transforms as cxfmapi
from .base import AbstractTransform
from .custom_types import CDict
from coordinax.internal import pack_nonuniform_unit, pack_uniform_unit

# A "point-like" input the entry funnel accepts. Faithful (each member and the
# union), so the normalizer methods below stay in plum's method cache.
# `guess_chart` accepts all of these directly (JAX and NumPy arrays, Quantities,
# QuantityMatrix, CDicts). A Python list is not an `ArrayLike`, so it never matches this
# union at all (see test_act_rejects_python_list).
PointLike: TypeAlias = ArrayLike | AbcQ | ul.QuantityMatrix | CDict


# ===================================================================
# Per-input-type representation default
#
# When ``rep`` is omitted the default depends only on the INPUT TYPE, not the
# operator: a bare array carries no unit information to infer a role from, so it
# defaults to ``point``; a ``QuantityMatrix`` (a raw metric/Jacobian container,
# even though it stores per-element units) has no semantic role to infer, so it
# also defaults to ``point``; a Quantity / CDict carries units, so the role is
# guessed from them.


# NB: the return type is ``Any`` on purpose. plum performs runtime return-type
# *conversion*, and a concrete `cxr.Representation` return trips the class's
# invariant generic (``cxr.point`` is ``Representation[PointGeometry, ...]``),
# while a `[Any, Any, Any]`-parametrized return breaks plum's converter. Both
# were verified to fail; ``Any`` is the only annotation that is correct at
# runtime and clean under the type checker.


@plum.dispatch
def _default_rep(x: ArrayLike, /) -> Any:
    return cxr.point


@plum.dispatch
def _default_rep(x: ul.QuantityMatrix, /) -> Any:
    return cxr.point


@plum.dispatch
def _default_rep(x: AbcQ, /) -> Any:
    return cxr.guess_rep(x)


@plum.dispatch
def _default_rep(x: CDict, /) -> Any:
    return cxr.guess_rep(x)


# ===================================================================
# Normalize-once entry funnel
#
# One pair of arity-3 / arity-4 methods fills in the missing ``chart`` (guessed
# once) and ``rep`` (per-type default), then redispatches to the arity-5 typed
# ``act`` — the per-(operator, input-type) JAX fast paths, or the arity-5
# generic coercion fallback for operators without a typed fast path. This
# replaces the former per-input-type × arity boilerplate (and its ``-1``
# precedence tier).
#
# The return type is ``Any``: these methods return the SAME container type they
# receive (Array->Array, Quantity->Quantity, ...), which the type system can't
# express, and because plum converts return values at runtime a `PointLike`
# union return coerces/mangles the result (verified against the test suite).


@plum.dispatch
def act(op: AbstractTransform, tau: Any, x: PointLike, /, **kw: Any) -> Any:
    """Infer the chart and representation, then apply the operator.

    A bare input is interpreted as the data for a `coordinax.Point` in its
    guessed (Cartesian) chart.

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

    A Quantity carries units, so its role is inferred from them:

    >>> q = u.Q([1, 0, 0], "km")
    >>> cxfm.act(R, None, q).round(3)
    Q([0., 1., 0.], 'km')

    """
    chart = cxc.guess_chart(x)
    return cxfmapi.act(op, tau, x, chart, _default_rep(x), **kw)


@plum.dispatch
def act(
    op: AbstractTransform,
    tau: Any,
    x: PointLike,
    chart: cxc.AbstractChart,
    /,
    **kw: Any,
) -> Any:
    """Infer the representation, then apply the operator.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxfm

    >>> R = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> q = u.Q([1, 0, 0], "km")
    >>> cxfm.act(R, None, q, cxc.cart3d).round(3)
    Q([0., 1., 0.], 'km')

    """
    return cxfmapi.act(op, tau, x, chart, _default_rep(x), **kw)


# ===================================================================
# Arity-5 typed fallbacks: coerce non-CDict inputs to a Cartesian CDict, act,
# and repack. Operators with a typed arity-5 fast path (rotate.py, translate.py,
# ...) override these by concrete-type specificity; these serve the rest.
#
# On Array(like) inputs


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

    The Array is interpreted as Cartesian point coordinates.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    >>> op = cx.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> x = jnp.asarray([1.0, 0.0, 0.0])
    >>> cx.act(op, None, x, cx.cart3d, cx.point).round(3)
    Array([0., 1., 0.], dtype=float64)

    """
    out = cxfmapi.act(op, tau, x, chart, rep.geom_kind, rep, **kw)
    return cast("Array", out)


# ===================================================================
# On Quantity inputs


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
    """Apply operator, routing through the CDict-based implementation.

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
# Precedence=2 so QuantityMatrix (a subclass of AbstractQuantity) prefers this typed
# path over the (SpecificTransform, AbstractQuantity) fast paths in rotate.py /
# translate.py / composed.py (precedence 0) AND over the Identity catch-all
# (precedence 1). Without it, e.g. (Composed, tau, QuantityMatrix) is ambiguous between
# (Composed, tau, AbcQ) and (AbstractTransform, tau, QuantityMatrix).


@plum.dispatch(precedence=2)  # ty: ignore[no-matching-overload]
def act(
    op: AbstractTransform,
    tau: Any,
    x: ul.QuantityMatrix,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    **kw: Any,
) -> ul.QuantityMatrix:
    """Apply an operator to a ``QuantityMatrix`` with explicit chart and rep.

    Routes through the CDict-based implementation, then repacks the result
    into a ``QuantityMatrix``.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.transforms as cxfm
    >>> import coordinax.representations as cxr
    >>> import unxts.linalg as ul

    >>> op = cxfm.Rotate.from_euler("z", u.Q(90, "deg"))
    >>> qm = ul.QuantityMatrix(
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
    return ul.QuantityMatrix(arr, unit=units)
