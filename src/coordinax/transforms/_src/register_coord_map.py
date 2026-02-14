"""Coordinate-basis tangent transformations (coord_transform) and vconvert glue."""

__all__: tuple[str, ...] = ()

import functools as ft

from collections.abc import Callable, Mapping
from typing import Any

import equinox as eqx
import jax
import jax.tree as jtu
import plum

import quaxed.numpy as jnp
from unxt.quantity import BareQuantity, is_any_quantity

import coordinax.api as cxapi
import coordinax.charts as cxc
import coordinax.roles as cxr
from coordinax._src.custom_types import OptUSys
from coordinax.api import CsDict

is_q_or_arr = lambda x: is_any_quantity(x) or eqx.is_array(x)  # noqa: E731


def _point_transform_with_usys(
    to_chart: cxc.AbstractChart[Any, Any],
    from_chart: cxc.AbstractChart[Any, Any],
    p: Mapping[str, Any],
    usys: OptUSys,
    /,
) -> CsDict:
    return cxapi.point_transform(to_chart, from_chart, p, usys=usys)


jac_point_fn_scalar = jax.jit(
    jax.jacfwd(_point_transform_with_usys, argnums=2), static_argnums=(0, 1, 3)
)


@ft.partial(jax.jit, inline=True)
def _inner_dot(inner: Any, vec: Any) -> Any:
    return jtu.reduce(
        jnp.add,
        jtu.map(jnp.multiply, inner, vec, is_leaf=is_q_or_arr),
        is_leaf=is_q_or_arr,
    )


@jax.jit
def _dot_jac_vec(jac: Any, vec: Any) -> dict[str, Any]:
    return {k: _inner_dot(inner, vec) for k, inner in jac.items()}


def _compute_jac(
    to_chart: cxc.AbstractChart[Any, Any],
    from_chart: cxc.AbstractChart[Any, Any],
    at: CsDict,
    usys: OptUSys,
    /,
) -> dict[str, dict[str, Any]]:
    """Compute and normalize the Jacobian of `point_transform` at `at`.

    `jax.jacfwd(point_transform)` over Quantity pytrees returns a structure like:
    `{to_k: Quantity({from_k: Quantity(dy, u_from)}, u_to)}`.

    Normalize this to `{to_k: {from_k: BareQuantity(dy, u_to/u_from)}}`.
    """
    jac = jac_point_fn_scalar(to_chart, from_chart, at, usys)

    out: dict[str, dict[str, Any]] = {}
    for out_k, out_v in jac.items():
        if is_any_quantity(out_v) and isinstance(out_v.value, Mapping):
            out[out_k] = {
                in_k: BareQuantity(v.value, out_v.unit / v.unit)
                for in_k, v in out_v.value.items()
            }
        else:
            out[out_k] = dict(out_v)
    return out


# =============================================================================
# vconvert (data-level) glue


@plum.dispatch
def vconvert(
    role: cxr.CoordDisp | cxr.CoordVel | cxr.CoordAcc,
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: CsDict,
    /,
    *_: CsDict,
    at: CsDict,
    usys: OptUSys = None,
) -> CsDict:
    """Convert coordinate-basis tangent components between charts."""
    return cxapi.coord_transform(to_chart, from_chart, x, at=at, usys=usys)


# =============================================================================
# coord_transform


@plum.dispatch
def coord_transform(
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    /,
) -> Callable[..., Any]:
    """Return a partial function for coordinate-basis tangent transformation."""
    return lambda *args, **kw: cxapi.coord_transform(to_chart, from_chart, *args, **kw)


@plum.dispatch
def coord_transform(
    to_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    from_chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: CsDict,
    /,
    *,
    at: CsDict,
    usys: OptUSys = None,
) -> CsDict:
    if to_chart is from_chart:
        return x

    if set(x.keys()) != set(from_chart.components):
        msg = "Data keys do not match from_chart components."
        raise ValueError(msg)

    from_chart.check_data(at)

    jac = _compute_jac(to_chart, from_chart, at, usys)
    out = _dot_jac_vec(jac, x)

    # Reshape the output to the broadcasted input shape (vmap-friendly)
    shapes = [jnp.shape(v) for v in x.values()]
    shape = jnp.broadcast_shapes(*shapes) if shapes else ()
    return jtu.map(lambda v: jnp.reshape(v, shape), out, is_leaf=is_q_or_arr)


@plum.dispatch
def coord_transform(
    to_chart: cxc.AbstractCartesianProductChart,  # type: ignore[type-arg]
    from_chart: cxc.AbstractCartesianProductChart,  # type: ignore[type-arg]
    x: CsDict,
    /,
    *,
    at: CsDict,
    usys: OptUSys = None,
) -> CsDict:
    if len(to_chart.factors) != len(from_chart.factors):
        msg = "Product charts must have the same number of factors."
        raise ValueError(msg)

    x_parts = from_chart.split_components(x)
    at_parts = from_chart.split_components(at)

    out_parts = tuple(
        cxapi.coord_transform(t, f, x_part, at=at_part, usys=usys)
        for t, f, x_part, at_part in zip(
            to_chart.factors, from_chart.factors, x_parts, at_parts, strict=True
        )
    )
    return to_chart.merge_components(out_parts)
