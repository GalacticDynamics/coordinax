# ruff: noqa: ERA001
"""Galilean coordinate transformations."""

__all__ = ["GalileanBoost"]


from dataclasses import replace
from typing import Any, Literal, final

import equinox as eqx
import jax.numpy as jnp
from plum import convert, dispatch

import quaxed.numpy as jnp
import unxt as u
from dataclassish.converters import Unless

from .base import AbstractGalileanOperator
from coordinax._src.operators.base import AbstractOperator
from coordinax._src.operators.identity import Identity
from coordinax._src.vectors.base import AbstractPos, AbstractVel
from coordinax._src.vectors.d3 import CartesianVel3D
from coordinax._src.vectors.d4 import FourVector


@final
class GalileanBoost(AbstractGalileanOperator):
    r"""Operator for Galilean boosts.

    The operation is given by:

    .. math::

        (t,\mathbf{x}) \mapsto (t, \mathbf{x} + \mathbf{v} t)

    where :math:`\mathbf{v}` is the boost velocity.

    Parameters
    ----------
    velocity : :class:`coordinax.vecs.AbstractVel`
        The boost velocity. This parameters uses
        :meth:`coordinax.vecs.CartesianVel3D.from_` to enable a variety of more
        convenient input types.

    Examples
    --------
    >>> import coordinax as cx

    >>> op = cx.ops.GalileanBoost.from_([1.0, 2.0, 3.0], "m/s")
    >>> op
    GalileanBoost(CartesianVel3D( ... ))

    >>> vec = cx.CartesianPos3D.from_([0.0, 0.0, 0.0], "m")

    >>> delta_t = u.Quantity(1.0, "s")
    >>> newvec, _ = op(vec, delta_t)
    >>> print(newvec)
    <CartesianPos3D (x[m], y[m], z[m])
        [1. 2. 3.]>

    In the context of frame transformations, a Galilean boost is treated as the
    velocity of the new frame relative to the old frame. This means that the
    transformation is the same as the inverse of the boost velocity. In other
    words,

    .. math::

        (t,\mathbf{x}) \mapsto (t, \mathbf{x} - \mathbf{v} t)

    This can be applied using the ``.inverse`` property.

    >>> vec_in_newframe, _ = op.inverse(vec, delta_t)
    >>> print(vec_in_newframe)
    <CartesianPos3D (x[m], y[m], z[m])
        [-1. -2. -3.]>

    """

    velocity: AbstractVel = eqx.field(
        converter=Unless(AbstractVel, CartesianVel3D.from_)
    )
    """The boost velocity.

    Unless given a :class:`coordinax.AbstractVel`, this parameter uses
    :meth:`coordinax.CartesianVel3D.from_` to enable a variety of more
    convenient input types. See :class:`coordinax.CartesianVel3D` for details.
    """

    # -----------------------------------------------------

    @property
    def is_inertial(self) -> Literal[True]:
        """Galilean boost is an inertial-frame preserving transform.

        Examples
        --------
        >>> import coordinax as cx

        >>> op = cx.ops.GalileanBoost.from_([1, 2, 3], "m/s")
        >>> op.is_inertial
        True

        """
        return True

    @property
    def inverse(self) -> "GalileanBoost":
        """The inverse of the operator.

        Examples
        --------
        >>> import coordinax as cx

        >>> op = cx.ops.GalileanBoost.from_([1, 2, 3], "m/s")
        >>> op.inverse
        GalileanBoost(CartesianVel3D( ... ))

        >>> print(op.inverse.velocity)
        <CartesianVel3D (d_x[m / s], d_y[m / s], d_z[m / s])
            [-1 -2 -3]>

        """
        return GalileanBoost(-self.velocity)

    # -----------------------------------------------------

    @AbstractOperator.__call__.dispatch
    def __call__(
        self: "GalileanBoost", q: AbstractPos, delta_t: u.Quantity["time"], /
    ) -> tuple[AbstractPos, u.Quantity["time"]]:
        """Apply the boost to the coordinates.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        Define a position:

        >>> q = cx.CartesianPos3D.from_([0, 0, 0], "m")

        Define a boost operator and the time interval to apply it:

        >>> op = cx.ops.GalileanBoost.from_([1, 2, 3], "m/s")
        >>> dt = u.Quantity(1, "s")

        >>> newq, _ = op(q, dt)

        The position is updated by the boost velocity times the time interval:

        >>> print(newq)
        <CartesianPos3D (x[m], y[m], z[m])
            [1 2 3]>

        """
        return q + self.velocity * delta_t, delta_t

    @AbstractOperator.__call__.dispatch
    def __call__(self: "GalileanBoost", v4: FourVector, /, **__: Any) -> FourVector:
        r"""Apply the boost to the coordinates.

        Recall that this is spatial-only, the time is invariant.

        The operation is given by:

        .. math::

            (t,\mathbf{x}) \mapsto (t, \mathbf{x} + \mathbf{v} t)

        Examples
        --------
        >>> import unxt as u

        """
        q, _ = self(v4.q, v4.t)
        return replace(v4, q=q)

    # -------------------------------------------
    # Arithmetic operations

    def __neg__(self: "GalileanBoost") -> "GalileanBoost":
        """Negate the rotation.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import coordinax as cx

        >>> op = cx.ops.GalileanBoost.from_([1, 0, 0], "m/s")
        >>> print((-op).velocity)
        <CartesianVel3D (d_x[m / s], d_y[m / s], d_z[m / s])
            [-1 0 0]>

        """
        return replace(self, velocity=-self.velocity)

    # -----------------------------------------------------
    # Python

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.velocity!r})"


# -----------------------------------------------------


@dispatch
def simplify_op(op: GalileanBoost, /, **kwargs: Any) -> GalileanBoost | Identity:
    """Simplify a boost operator.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    An operator with real effect cannot be simplified:

    >>> op = cx.ops.GalileanBoost.from_([1, 0, 0], "m/s")
    >>> cx.ops.simplify_op(op)
    GalileanBoost(CartesianVel3D( ... ))

    An operator with no effect can be simplified:

    >>> op = cx.ops.GalileanBoost.from_([0, 0, 0], "m/s")
    >>> cx.ops.simplify_op(op)
    Identity()

    """
    # Check if the velocity is zero.
    if jnp.allclose(convert(op.velocity, u.Quantity).value, jnp.zeros((3,)), **kwargs):
        return Identity()
    return op


@dispatch
def simplify_op(op1: GalileanBoost, op2: GalileanBoost) -> GalileanBoost:
    """Combine two boosts into a single boost.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    >>> op1 = cxo.GalileanBoost.from_([1, 0, 0], "m/s")
    >>> op2 = cxo.GalileanBoost.from_([0, 1, 0], "m/s")

    >>> op3 = cxo.simplify_op(op1, op2)
    >>> op3
    GalileanBoost(CartesianVel3D( ... ))

    >>> op3.velocity == op1.velocity + op2.velocity
    Array(True, dtype=bool)

    """
    return GalileanBoost(op1.velocity + op2.velocity)
