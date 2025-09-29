"""transformations by coordinate addition."""

__all__ = ("Translate",)


from typing import Any, Literal, final

import equinox as eqx
from plum import convert, dispatch

import quaxed.numpy as jnp
import unxt as u

from .base import AbstractOperator
from coordinax._src.operators.api import invert
from coordinax._src.operators.base import AbstractOperator
from coordinax._src.operators.identity import Identity
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.d3 import AbstractPos3D


@final
class Translate(AbstractOperator):
    r"""Operator for spatio-temporal translations.

    The coordinate transform is given by:

    .. math::

        (t,\mathbf{x}) \mapsto (t+s, \mathbf{x} + \mathbf {a})

    where :math:`a \in R^3` and :math:`s \in R`.  Therefore for a potential
    :math:`\Phi(t,\mathbf{x})` in the translated frame the potential is given by
    the subtraction of the translation.

    Parameters
    ----------
    delta_t
        The time translation.
    delta_q
        The spatial translation vector.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    We can then create a translation operator:

    >>> op = cx.ops.Translate.from_([1.0, 2.0, 3.0, 4.0], "km")
    >>> op
    Translate(
      delta_t=Quantity(f32[], unit='s'),
      delta_q=CartesianPos3D( ... )
    )

    Note that the translation is a `unxt.Quantity` and a
    `coordinax.vecs.AbstractPos`, which was constructed from a 1D array, using
    :meth:`coordinax.vecs.AbstractPos.from_`.  We can also construct it
    directly, which allows for other vector types.

    >>> qshift = cx.SphericalPos(r=u.Quantity(1.0, "km"),
    ...                          theta=u.Quantity(jnp.pi/2, "rad"),
    ...                          phi=u.Quantity(0, "rad"))
    >>> op = cx.ops.Translate(u.Quantity(1.0, "Gyr"), qshift)
    >>> op
    Translate(
      delta_t=Quantity(weak_f32[], unit='Gyr'),
      delta_q=SphericalPos( ... )
    )

    Translation operators can be applied to `coordinax.vecs.AbstractPos` and
    `unxt.Quantity`:

    >>> q = cx.CartesianPos3D.from_([0, 0, 0], "km")
    >>> t = u.Quantity(0, "Gyr")
    >>> newt, newq = op(t, q)
    >>> newq.x
    Quantity(Array(1., dtype=float32, ...), unit='km')
    >>> newt
    Quantity(Array(1., dtype=float32, ...), unit='Gyr')

    """

    delta: AbstractPos = eqx.field()
    """The translation."""

    # -------------------------------------------

    @property
    def is_inertial(self) -> Literal[True]:
        """Galilean translation is an inertial-frame preserving transformation.

        Examples
        --------
        >>> import coordinax as cx

        >>> op = cx.ops.Translate.from_([0, 1, 1, 1], "km")

        >>> op.is_inertial
        True

        """
        return True

    @property
    def inverse(self) -> "Translate":
        """The inverse of the operator.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> qshift = cx.CartesianPos3D.from_([1, 1, 1], "km")
        >>> op = cx.ops.Translate(u.Quantity(1, "Gyr"), qshift)

        >>> print(op.inverse)
        Translate(
            delta_t=Quantity(-1, unit='Gyr'),
            delta_q=<CartesianPos3D: (x, y, z) [km]
                [-1 -1 -1]>
        )

        """
        return Translate(invert(self.delta))


@dispatch
def operate(
    self: Translate,
    t: u.Quantity["time"],
    x: AbstractPos3D,
    /,
    **__: Any,
) -> tuple[u.Quantity["time"], AbstractPos3D]:
    """Apply the translation to the coordinates.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx
    >>> import coordinax.ops as cxo

    Explicitly construct the translation operator:

    >>> qshift = cx.CartesianPos3D.from_([1, 1, 1], "km")
    >>> tshift = u.Quantity(1, "Gyr")
    >>> op = cx.ops.Translate(tshift, qshift)

    Construct a vector to translate

    >>> q = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> t = u.Quantity(1, "Gyr")
    >>> newt, newq = op(t, q)

    >>> newq.x
    Quantity(Array(2, dtype=int32), unit='km')

    >>> newt
    Quantity(Array(2, dtype=int32, ...), unit='Gyr')

    """
    return x + self.delta


# -------------------------------------------


@dispatch
def simplify_op(op: Translate, /, **kwargs: Any) -> Translate | Identity:
    """Simplify a Galilean translation operator.

    Examples
    --------
    >>> import coordinax.ops as cxo

    An operator with real effect cannot be simplified:

    >>> op = cxo.Translate.from_([3e8, 1, 0, 0], "m")
    >>> cxo.simplify_op(op)
    Translate(
      delta_t=Quantity(f32[], unit='m s / km'),
      delta_q=CartesianPos3D( ... )
    )

    An operator with no effect can be simplified:

    >>> op = cxo.Translate.from_([0, 0, 0, 0], "m")
    >>> cxo.simplify_op(op)
    Identity()

    """
    # Check if the translation is zero.
    q = convert(op.delta_q, u.Quantity).value
    if jnp.allclose(op.delta_t.value, 0, **kwargs) and jnp.allclose(q, 0, **kwargs):
        return Identity()
    # TODO: Check if the translation is purely spatial.

    return op


# TODO: show op3.translation = op1.translation + op2.translation
@dispatch
def simplify_op(op1: Translate, op2: Translate, /) -> Translate:
    """Combine two translations into a single translation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> qshift = cx.CartesianPos3D.from_([1, 0, 0], "km")
    >>> tshift = u.Quantity(1, "Gyr")
    >>> op1 = cx.ops.Translate(tshift, qshift)

    >>> qshift = cx.CartesianPos3D.from_([0, 1, 0], "km")
    >>> op2 = cx.ops.Translate(tshift, qshift)

    >>> op3 = cx.ops.simplify_op(op1, op2)
    >>> op3
    Translate(
      delta_t=Quantity(weak_i32[], unit='Gyr'),
      delta_q=CartesianPos3D( ... )
    )

    """
    return Translate(op1.delta_t + op2.delta_t, op1.delta_q + op2.delta_q)
