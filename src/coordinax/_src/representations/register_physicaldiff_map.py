"""Conversion functions for vector representations."""

__all__ = ()

import jax.tree as jtu
import plum

import quaxed.numpy as jnp
import unxt as u

from . import api, euclidean as r
from .custom_types import PDict
from .frames import frame_to_cart, pullback, pushforward
from .roles import Acc, Vel


@plum.dispatch
def vconvert(
    role: Vel | Acc,
    to_rep: r.AbstractRep,
    from_rep: r.AbstractRep,
    p_dif: PDict,
    p_pos: PDict,
    /,
    *_: PDict,
) -> PDict:
    """Convert a position vector from one representation to another."""
    # Convert using the representation conversion
    return api.diff_map(to_rep, from_rep, p_dif, p_pos)


def _pack_uniform_unit(
    p: PDict, keys: tuple[str, ...]
) -> tuple[jnp.ndarray, u.AbstractUnit]:
    """Pack dict-of-quantities into an array, converting all entries to a common unit."""
    # Choose a reference unit from the first key.
    ref_unit = p[keys[0]].unit
    vals = [u.uconvert(ref_unit, p[k]).value for k in keys]
    return jnp.stack(vals, axis=-1), ref_unit


def _unpack_with_unit(
    vals: jnp.ndarray, unit: u.AbstractUnit, keys: tuple[str, ...]
) -> PDict:
    """Unpack an array into dict-of-quantities with a shared unit."""
    return {k: u.Quantity(vals[..., i], unit) for i, k in enumerate(keys)}


@plum.dispatch
def diff_map(
    to_rep: r.AbstractRep, from_rep: r.AbstractRep, p_dif: PDict, p_pos: PDict, /
) -> PDict:
    """Transform **physical** differential components by orthonormal-frame change.

    This treats `p_dif` as components of a physical vector (velocity or acceleration)
    expressed in the orthonormal frame associated with `from_rep`, evaluated at the
    position `p_pos` (given in `from_rep` coordinates).

    The same geometric vector is re-expressed in the orthonormal frame associated with
    `to_rep` at the same physical point, using the target metric.

    Notes
    -----
    This is *not* the correct rule for transforming coordinate time-derivatives
    (e.g. dtheta/dt).

    """
    # Convert the position into the target rep so we can evaluate its frame at
    # the same point.
    p_pos_to = api.coord_map(to_rep, from_rep, p_pos)

    # Orthonormal frames in Cartesian components at the same physical point.
    B_from = frame_to_cart(from_rep, p_pos)
    B_to = frame_to_cart(to_rep, p_pos_to)

    # Pack vector components (uniform unit: speed or acceleration) in rep component order.
    from_keys = tuple(from_rep.components)
    to_keys = tuple(to_rep.components)

    v_from, unit = _pack_uniform_unit(p_dif, from_keys)

    # v_cart = B_from @ v_from
    v_cart = pushforward(B_from, v_from)

    # Pull back into the target frame using the target metric.
    v_to = pullback(api.metric_of(to_rep), B_to, v_cart)

    # Unpack to dict with shared unit
    out = _unpack_with_unit(v_to, unit, to_keys)

    # Reshape outputs to broadcast shape of inputs
    shape = jnp.broadcast_shapes(*[v.shape for v in p_dif.values()])
    return jtu.map(lambda x: jnp.reshape(x, shape), out)
