"""Core operator API functions.

This module defines helpers for operator implementations.
"""

__all__: tuple[str, ...] = (
    "act_array_via_cdict",
    "act_quantity_via_cdict",
    "is_componentwise_offset",
    "is_flat_chart",
    "require_matching_keys",
)

from collections.abc import Iterable
from typing import Any, Final

import jax.numpy as jnp

import unxt as u
import unxts.linalg as ul

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinaxs.api.charts as cxcapi
import coordinaxs.api.transforms as cxfmapi
from coordinax._src.exceptions import NoGlobalCartesianChartError
from coordinax.internal import pack_uniform_unit

DMLS: Final = u.unit("")


def is_flat_chart(chart: Any, /) -> bool:
    """Whether ``chart`` is a Cartesian-type chart (its own canonical Cartesian).

    In such charts a componentwise offset IS a translation of the flat
    ambient space (Jacobian = identity, no base-point dependence). In any
    other chart an offset must be pushed through the chart Jacobian at the
    point, so additive fast paths do not apply.

    A chart with no global Cartesian chart (e.g. ``PoincarePolar6D``) is not
    flat: this predicate returns `False` rather than propagating
    `~coordinax.charts.NoGlobalCartesianChartError`.
    """
    try:
        cart = chart.cartesian
    except NoGlobalCartesianChartError:
        return False
    return isinstance(chart, type(cart))


def is_componentwise_offset(op: Any, chart: Any, /) -> bool:
    """Whether an additive offset acts componentwise on data in ``chart``.

    True for fibre-only offsets (ladder order k >= 1 — their point action is
    the identity, so the componentwise rule is definitional), and for k = 0
    offsets whose ``delta`` and data share the same Cartesian-type (flat)
    chart (a true ambient translation). Everything else is base-point
    dependent and must go through the generic engine.

    This is THE routing predicate for the additive family — `act`,
    `pushforward`, and `act_jet` must all use it so the fast paths stay
    provably consistent with the generic prolongation.
    """
    k = getattr(op, "semantic_kind", cxr.dpl).order
    return k != 0 or (chart == op.chart and is_flat_chart(chart))


def require_matching_keys(
    actual: Iterable[str], expected: Iterable[str], message: str, /
) -> None:
    """Raise ``TypeError(message + missing/unexpected keys)`` if keys differ.

    Shared by `act`, `pushforward`, and `prolong` so component-mismatch errors
    report the missing and unexpected keys in one consistent format.
    """
    got, exp = set(actual), set(expected)
    if got != exp:
        miss, extra = sorted(exp - got), sorted(got - exp)
        raise TypeError(
            message
            + (f"; missing {miss}" if miss else "")
            + (f"; unexpected {extra}" if extra else "")
            + "."
        )


# ===================================================================
# Coerce-act-repack funnels
#
# The CDict methods are the reference implementation of every operator: they
# cover the full (representation, semantic kind) ladder. A Quantity or a bare
# array is therefore best served by converting to a CDict, acting there, and
# repacking — which is what these two helpers do. They are shared by the
# generic arity-5 fallbacks in `register_apply` and by the typed fast paths
# (e.g. `Translate`) for the cells their fast path does not cover, so the two
# can never disagree.


def act_quantity_via_cdict(
    op: Any, tau: Any, x: Any, chart: Any, rep: Any, /, **kw: Any
) -> Any:
    """Act on a `unxt.AbstractQuantity` through its Cartesian `CDict`.

    The result is repacked into a Quantity, which requires the acted-on
    components to share a unit (they do in a Cartesian chart).
    """
    v = cxc.cdict(x, chart)
    nv = cxfmapi.act(op, tau, v, chart, rep, **kw)
    value, unit = pack_uniform_unit(nv, keys=chart.components)  # ty: ignore[no-matching-overload]
    return u.Q(value, unit)


def act_array_via_cdict(
    op: Any, tau: Any, x: Any, chart: Any, rep: Any, /, *, usys: Any = None, **kw: Any
) -> Any:
    """Act on a bare array through a `CDict`, taking its units from ``usys``.

    A bare array carries no units, so ``usys`` supplies them: each component
    is read in the unit that ``usys`` gives for that component's dimension
    under ``rep`` (``chart.coord_dimensions`` differentiated ``rep``'s ladder
    order). The result is written back in those same units, so an array in is
    an array out.
    """
    if usys is None:
        msg = (
            f"{type(op).__name__} requires 'usys' to act on a bare array: the "
            "array carries no units, so they are read from the unit system. "
            "Pass usys=..., or use a Quantity / component dict / typed vector, "
            "which carry their own units."
        )
        raise TypeError(msg)

    # `coord_dimensions` answers for the *coordinate* basis, so it is only a
    # correct reading of a bare array when that is the basis in play. In a
    # physical (orthonormal) basis every component of a velocity is a speed,
    # while `coord_dimensions` still reports angular speed for the angular
    # components of a curvilinear chart -- which would silently mis-unit the
    # array. Refuse rather than guess; a unit-carrying container says it.
    if not isinstance(rep.basis, cxr.NoBasis | cxr.CoordinateBasis):
        msg = (
            f"{type(op).__name__} cannot act on a bare array in the "
            f"{type(rep.basis).__name__}: units for a bare array are read from "
            "the chart's coordinate dimensions, which describe the coordinate "
            "basis. Pass a Quantity, a QuantityMatrix, or a typed vector, which "
            "carry their own per-component units."
        )
        raise TypeError(msg)

    dims = rep.semantic_kind.coord_dimensions(chart)
    units = tuple(DMLS if d is None else usys[d] for d in dims)

    x_arr = jnp.asarray(x)
    # Check the shape here rather than letting `QuantityMatrix` reject it: it
    # would, but in its own vocabulary ("value trailing shape (2,) does not
    # match the unit structure"), which describes a type the caller never
    # named. Ask about the last axis rather than indexing for it -- a 0-D
    # array has none, and that is as wrong a shape as a mismatched one.
    shape = jnp.shape(x_arr)
    if not shape or shape[-1] != len(chart.components):
        got = shape[-1] if shape else "no axes"
        msg = (
            f"act for {type(op).__name__}: last axis of x is {got}, but "
            f"{type(chart).__name__} has {len(chart.components)} components."
        )
        raise ValueError(msg)

    v = cxc.cdict(ul.QuantityMatrix(x_arr, unit=units), chart)
    nv = cxfmapi.act(op, tau, v, chart, rep, usys=usys, **kw)
    return cxcapi.carray(nv, chart.components, usys).value  # ty: ignore[unresolved-attribute]


def _unnormalisable(norm: Any, /) -> Any:
    """Whether ``v / norm`` fails to be a unit vector, for ``norm = |v|``.

    True unless the norm is finite and strictly positive, which is exactly the
    precondition: a norm is NaN iff a component of ``v`` is, ``inf`` iff a
    component is, and ``0`` iff ``v`` is.
    """
    return ~((norm > 0) & jnp.isfinite(norm))
