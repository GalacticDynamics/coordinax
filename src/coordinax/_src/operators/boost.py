"""Galilean coordinate transformations."""

__all__ = ["VelocityBoost"]


from dataclasses import replace
from typing import Any, Literal, final

import equinox as eqx
import jax.numpy as jnp
from plum import convert, dispatch

import quaxed.numpy as jnp
import unxt as u
from dataclassish.converters import Unless

from .base import AbstractOperator
from .identity import Identity
from coordinax._src.vectors.base import AbstractPos, AbstractVel
from coordinax._src.vectors.d3 import CartesianVel3D


@final
class VelocityBoost(AbstractOperator):
    r"""Operator for an instantaneous velocity boost.

    The operation is given by:

    .. math::

        (\mathbf{v}) \mapsto (\mathbf{v} + \Delta\mathbf{v})

    where :math:`\Delta\mathbf{v}` is the boost velocity.

    Parameters
    ----------
    velocity : :class:`coordinax.vecs.AbstractVel`
        The boost velocity. This parameters uses
        :meth:`coordinax.vecs.CartesianVel3D.from_` to enable a variety of more
        convenient input types.

    Examples
    --------
    >>> import coordinax as cx

    >>> op = cx.ops.VelocityBoost.from_([1.0, 2.0, 3.0], "m/s")
    >>> op
    VelocityBoost(CartesianVel3D( ... ))

    >>> q = cx.CartesianPos3D.from_([0.0, 0.0, 0.0], "m")
    >>> op(q) is q
    True

    >>> p = cx.CartesianVel3D.from_([0., 0, 0], "m/s")
    >>> op(p) == p + op.velocity
    Array(True, dtype=bool)

    """

    velocity: AbstractVel = eqx.field(
        converter=Unless(AbstractVel, CartesianVel3D.from_)
    )
    """The boost velocity.

    Unless given a :class:`coordinax.AbstractVel`, this parameter uses
    :meth:`coordinax.CartesianVel3D.from_` to enable a variety of more
    convenient input types.
    """

    # -----------------------------------------------------

    @property
    def is_inertial(self) -> Literal[True]:
        """Galilean boost is an inertial-frame preserving transform.

        Examples
        --------
        >>> import coordinax as cx

        >>> op = cx.ops.VelocityBoost.from_([1, 2, 3], "m/s")
        >>> op.is_inertial
        True

        """
        return True

    @property
    def inverse(self) -> "VelocityBoost":
        """The inverse of the operator.

        Examples
        --------
        >>> import coordinax as cx

        >>> op = cx.ops.VelocityBoost.from_([1, 2, 3], "m/s")
        >>> op.inverse
        VelocityBoost(CartesianVel3D( ... ))

        >>> print(op.inverse.velocity)
        <CartesianVel3D (d_x[m / s], d_y[m / s], d_z[m / s])
            [-1 -2 -3]>

        """
        return VelocityBoost(-self.velocity)

    # -----------------------------------------------------

    @AbstractOperator.__call__.dispatch  # type: ignore[attr-defined, misc]
    def __call__(self: "VelocityBoost", p: AbstractVel, /) -> AbstractVel:
        """Apply the boost to the coordinates.

        This does nothing to the position, as the boost is to the velocity only.

        Examples
        --------
        >>> import coordinax as cx

        >>> op = cx.ops.VelocityBoost.from_([1, 2, 3], "m/s")

        >>> p = cx.CartesianVel3D.from_([0, 0, 0], "m/s")
        >>> print(op(p))
        <CartesianVel3D (d_x[m / s], d_y[m / s], d_z[m / s])
            [1 2 3]>

        """
        return p + self.velocity

    # -------------------------------------------
    # Arithmetic operations

    def __neg__(self: "VelocityBoost") -> "VelocityBoost":
        """Negate the rotation.

        Examples
        --------
        >>> import quaxed.numpy as jnp
        >>> import coordinax as cx

        >>> op = cx.ops.VelocityBoost.from_([1, 0, 0], "m/s")
        >>> print((-op).velocity)
        <CartesianVel3D (d_x[m / s], d_y[m / s], d_z[m / s])
            [-1 0 0]>

        """
        return replace(self, velocity=-self.velocity)

    # -----------------------------------------------------
    # Python

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.velocity!r})"


# ======================================================================
# More call dispatch


@AbstractOperator.__call__.dispatch
def call(
    self: VelocityBoost, q: AbstractPos, p: AbstractVel, /
) -> tuple[AbstractPos, AbstractVel]:
    r"""Apply the boost to the coordinates.

    This does nothing to the position, as the boost is to the velocity only.

    Examples
    --------
    >>> import coordinax as cx

    >>> op = cx.ops.VelocityBoost.from_([1, 2, 3], "m/s")

    >>> q = cx.CartesianPos3D.from_([0, 0, 0], "m")
    >>> p = cx.CartesianVel3D.from_([0, 0, 0], "m/s")
    >>> newq, newp = op(q, p)
    >>> print(newq, newp, sep="\n")
    <CartesianPos3D (x[m], y[m], z[m])
        [0 0 0]>
    <CartesianVel3D (d_x[m / s], d_y[m / s], d_z[m / s])
        [1 2 3]>

    """
    pc = p.vconvert(type(self.velocity), q).uconvert(self.velocity.units)
    newp = pc + self.velocity
    return q, newp.vconvert(type(p), q)


@AbstractOperator.__call__.dispatch
def call(
    self: VelocityBoost, q: u.Quantity["length"], p: u.Quantity["speed"], /
) -> tuple[u.Quantity["length"], u.Quantity["speed"]]:
    r"""Apply the boost to the coordinates.

    This does nothing to the position, as the boost is to the velocity only.

    Examples
    --------
    >>> import coordinax as cx

    >>> op = cx.ops.VelocityBoost.from_([1, 2, 3], "m/s")

    >>> q = u.Quantity([0., 0, 0], "m")
    >>> p = u.Quantity([0., 0, 0], "m/s")
    >>> newq, newp = op(q, p)
    >>> print(newq, newp, sep="\n")
    Quantity['length'](Array([0., 0., 0.], dtype=float32), unit='m')
    Quantity['speed'](Array([1., 2., 3.], dtype=float32), unit='m / s')

    """
    pvec = CartesianVel3D.from_(p)
    newpvec = pvec + self.velocity
    return q, convert(newpvec, u.Quantity)


@AbstractOperator.__call__.dispatch(precedence=-1)
def call(self: VelocityBoost, q: AbstractPos, /) -> AbstractPos:
    """Apply the boost to the coordinates.

    This does nothing to the position, as the boost is to the velocity only.

    Examples
    --------
    >>> import coordinax as cx

    >>> op = cx.ops.VelocityBoost.from_([1, 2, 3], "m/s")

    >>> q = cx.CartesianPos3D.from_([0, 0, 0], "m")
    >>> op(q) is q
    True

    """
    return q


@AbstractOperator.__call__.dispatch
def call(
    self: VelocityBoost, q: AbstractPos, t: u.Quantity["time"], /
) -> tuple[AbstractPos, u.Quantity["time"]]:
    """Apply the boost to the coordinates.

    This does nothing to the position, as the boost is to the velocity only.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> op = cx.ops.VelocityBoost.from_([1, 2, 3], "m/s")

    >>> q = cx.CartesianPos3D.from_([0, 0, 0], "m")
    >>> t = u.Quantity(1, "s")

    >>> newq, newt = op(q, t)
    >>> newq is q, newt is t
    (True, True)

    """
    return q, t


# ======================================================================
# Simplification


@dispatch
def simplify_op(op: VelocityBoost, /, **kwargs: Any) -> VelocityBoost | Identity:
    """Simplify a boost operator.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    An operator with real effect cannot be simplified:

    >>> op = cx.ops.VelocityBoost.from_([1, 0, 0], "m/s")
    >>> cx.ops.simplify_op(op)
    VelocityBoost(CartesianVel3D( ... ))

    An operator with no effect can be simplified:

    >>> op = cx.ops.VelocityBoost.from_([0, 0, 0], "m/s")
    >>> cx.ops.simplify_op(op)
    Identity()

    """
    # Check if the velocity is zero.
    if jnp.allclose(convert(op.velocity, u.Quantity).value, jnp.zeros((3,)), **kwargs):
        return Identity()
    return op


@dispatch
def simplify_op(op1: VelocityBoost, op2: VelocityBoost) -> VelocityBoost:
    """Combine two boosts into a single boost.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    >>> op1 = cxo.VelocityBoost.from_([1, 0, 0], "m/s")
    >>> op2 = cxo.VelocityBoost.from_([0, 1, 0], "m/s")

    >>> op3 = cxo.simplify_op(op1, op2)
    >>> op3
    VelocityBoost(CartesianVel3D( ... ))

    >>> op3.velocity == op1.velocity + op2.velocity
    Array(True, dtype=bool)

    """
    return VelocityBoost(op1.velocity + op2.velocity)
