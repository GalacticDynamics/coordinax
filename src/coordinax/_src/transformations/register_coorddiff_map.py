"""Conversion functions for vector representations."""

__all__: tuple[str, ...] = ()

import functools as ft

from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax
import jax.tree as jtu
import plum

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import BareQuantity, is_any_quantity

from coordinax._src.charts import AbstractChart
from coordinax._src.custom_types import CsDict

jac_pos_fn_scalar = jax.jit(
    jax.jacfwd(coord_transform, argnums=2), static_argnums=(0, 1)
)


def jac_pos_fn(
    to_chart: AbstractChart[Any, Any],
    from_chart: AbstractChart[Any, Any],
    data: Mapping[str, Any],
    /,
) -> CsDict:
    return jax.vmap(jac_pos_fn_scalar, in_axes=(None, None, 0))(
        to_chart, from_chart, jtu.map(jnp.atleast_1d, data)
    )


def compute_jac(
    to_chart: AbstractChart[Any, Any],
    from_chart: AbstractChart[Any, Any],
    data: Mapping[str, Any],
    /,
) -> dict[str, dict[str, u.AbstractQuantity]]:
    """Compute the Jacobian of the transformation."""
    # Compute the Jacobian of the transformation.
    # NOTE: this is using Quantities. Using raw arrays is ~20x faster.
    jac = jac_pos_fn(to_chart, from_chart, data)
    # Restructure the Jacobian:
    # from: ``{to_k: Quantity({from_k: Quantity(dto/dfrom, u_from)}, u_to)}``
    # to  : ``{to_k: {from_k: Quantity(dto/dfrom, u_to/u_from)}}``.
    jac = {
        out_k: {
            k: BareQuantity(v.value, out_v.unit / v.unit)
            for k, v in out_v.value.items()
        }
        for out_k, out_v in jac.items()
    }
    return jac  # type: ignore[return-value]  # noqa: RET504


is_q_or_arr = lambda x: is_any_quantity(x) or eqx.is_array(x)  # noqa: E731


@ft.partial(jax.jit, inline=True)
def inner_dot(
    inner: dict[str, u.AbstractQuantity],
    vec: dict[str, u.AbstractQuantity],
) -> u.AbstractQuantity:
    """Dot product of two dicts.

    This is a helper function for the `dot_jac_vec` function.

    Parameters
    ----------
    inner
        The first dict.
        The structure is ``{from_k: Quantity(v, unit)}``.
    vec
        The second dict.
        The structure is ``{from_k: Quantity(v, unit)}``.

    """
    return jtu.reduce(
        jnp.add,
        jtu.map(jnp.multiply, inner, vec, is_leaf=is_q_or_arr),
        is_leaf=is_q_or_arr,
    )


@jax.jit
def dot_jac_vec(
    jac: dict[str, dict[str, u.AbstractQuantity]], vec: dict[str, u.AbstractQuantity]
) -> dict[str, u.AbstractQuantity]:
    """Dot product of a Jacobian dict and a vector dict.

    This is a helper function for the `coord_transform` function.

    Parameters
    ----------
    jac
        The Jacobian of the transformation.
        The structure is ``{to_k: {from_k: Quantity(v, unit)}}``.
    vec
        The vector to transform.
        The structure is ``{from_k: Quantity(v, unit)}``.

    Examples
    --------
    >>> import unxt as u

    >>> J = {"r": {"x": u.Q(1, ""), "y": u.Q(2, "")},
    ...      "phi": {"x": u.Q(3, "rad/km"), "y": u.Q(4, "rad/km")}}

    >>> v = {"x": u.Q(1, "km"), "y": u.Q(2, "km")}

    >>> dot_jac_vec(J, v)
    {'phi': Quantity(Array(11, dtype=int64, ...), unit='rad'),
     'r': Quantity(Array(5, dtype=int64, ...), unit='km')}

    """
    # TODO: rewrite this by separating the units and the values
    return {k: inner_dot(inner, vec) for k, inner in jac.items()}


@plum.dispatch
def coord_transform(
    to_chart: AbstractChart,  # type: ignore[type-arg]
    from_chart: AbstractChart,  # type: ignore[type-arg]
    p_dif: CsDict,
    p_pos: CsDict,
    /,
) -> CsDict:
    # Compute the Jacobian of the position transformation.
    # The position is assumed to be in the type required by the differential to
    # construct the Jacobian. E.g. for CartVel1D -> RadialVel, we need the
    # Jacobian of the CartPos1D -> RadialPos transform.
    jac = compute_jac(to_chart, from_chart, p_pos)  # type: ignore[arg-type]

    # Transform the differential by dotting with the Jacobian.
    to_p_dif = dot_jac_vec(jac, p_dif)

    # Reshape the output to the shape of the input
    shape = jnp.broadcast_shapes(*[v.shape for v in p_dif.values()])
    to_p_dif = jtu.map(lambda x: jnp.reshape(x, shape), to_p_dif)

    return to_p_dif  # noqa: RET504
