"""Boost (velocity-offset) operator."""

__all__ = ("Boost",)

from typing import Any, cast, final

import jax.tree as jtu
import plum

import quaxed.numpy as jnp
import unxt as u

import coordinax.charts as cxc
import coordinax.representations as cxr
from .add import AbstractAdd
from .base import materialize_transform
from coordinax.internal.custom_types import CDict, OptUSys
from coordinax.transforms._src import groups


@final
class Boost(AbstractAdd):
    r"""Operator for boosting velocities (Galilean velocity offset).

    A Boost operator adds a constant velocity offset :math:`\Delta v` to the
    velocity components of a phase-space point.  It acts trivially on position
    (point) data, displacement data, and acceleration data.

    Formally, in a Cartesian chart:
    :math:`B_{\Delta v}:\; \dot{x} \mapsto \dot{x} + \Delta v`.

    Parameters
    ----------
    delta : CDict | Callable[[tau], CDict]
        The velocity offset to apply.
        If callable, will be evaluated at the time parameter ``tau``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import coordinax.transforms as cxfm

    Create a boost operator:

    >>> delta_v = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
    >>> boost = cxfm.Boost(delta_v, chart=cxc.cart3d)
    >>> boost
    Boost({'x': 1.0, 'y': 0.0, 'z': 0.0}, chart=Cart3D())

    The inverse negates the velocity offset:

    >>> boost.inverse
    Boost({'x': -1.0, 'y': -0.0, 'z': -0.0}, chart=Cart3D())

    """

    # delta, chart, and right_add inherited from AbstractAdd
    @classmethod
    def groups(cls) -> frozenset[type]:
        """Return the groups to which this map belongs."""
        del cls
        return frozenset((groups.DiffeomorphismGroup,))


# ============================================================================
# act


@plum.dispatch
def act(
    op: Boost,
    tau: Any,
    x: CDict,
    chart: cxc.AbstractChart,
    rep: cxr.Representation,
    /,
    usys: OptUSys = None,
    **kw: Any,
) -> CDict:
    """Apply Boost to a component dictionary.

    A Boost only affects velocity vectors; it is the identity for points,
    displacements, and accelerations.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import coordinax.charts as cxc
    >>> import coordinax.representations as cxr
    >>> import coordinax.transforms as cxfm

    Boost acts on velocity (shifts components):

    >>> boost = cxfm.Boost(
    ...     {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)},
    ...     chart=cxc.cart3d,
    ... )
    >>> v = {"x": jnp.array(2.0), "y": jnp.array(3.0), "z": jnp.array(0.0)}
    >>> cxfm.act(boost, None, v, cxc.cart3d, cxr.coord_vel)
    {'x': Array(3., dtype=float64, ...), 'y': Array(3., dtype=float64, ...),
     'z': Array(0., dtype=float64, ...)}

    Boost is identity for positions:

    >>> p = {"x": jnp.array(1.0), "y": jnp.array(2.0), "z": jnp.array(3.0)}
    >>> cxfm.act(boost, None, p, cxc.cart3d, cxr.point)
    {'x': Array(1., dtype=float64, ...), 'y': Array(2., dtype=float64, ...),
     'z': Array(3., dtype=float64, ...)}

    """
    del kw, usys

    op_eval = materialize_transform(op, tau)

    if isinstance(rep.semantic_kind, cxr.Velocity):
        if chart != op_eval.chart:
            msg = (
                "Boost requires the input chart to match the boost chart. "
                f"Got input chart={chart!r} and boost chart={op_eval.chart!r}."
            )
            raise ValueError(msg)
        return cast(
            "CDict",
            jtu.map(
                jnp.add,
                *((x, op_eval.delta) if op_eval.right_add else (op_eval.delta, x)),
                is_leaf=u.quantity.is_any_quantity,
            ),
        )

    # Identity for points, displacements, and accelerations.
    return x
