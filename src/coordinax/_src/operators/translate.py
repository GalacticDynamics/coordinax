"""Coordinate translations."""

__all__ = ["Translate"]


from typing import Any, Literal, cast, final

import equinox as eqx
from plum import convert, dispatch

import quaxed.numpy as jnp
import unxt as u
from dataclassish.converters import Unless

from coordinax._src.operators.base import AbstractOperator
from coordinax._src.operators.identity import Identity
from coordinax._src.vectors import api
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.vectors.d3 import AbstractPos3D


@final
class Translate(AbstractOperator):
    r"""Operator for spatial and temporal translations.

    The coordinate transform is given by:

    .. math::

        (t,\mathbf{x}) \mapsto (t+s, \mathbf{x} + \mathbf {a})

    where :math:`a \in R^3` and :math:`s \in R`.

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

    TODO: actually this isn't right - it does some four-vector magic I don't like!
    Here, no time translation is specified, so it defaults to zero (in seconds, but the
    unit does not matter).

    Note that position translation is a `coordinax.vecs.CartesianPos3D`, which was
    constructed from a 1D array, using :meth:`coordinax.vecs.AbstractPos.from_`.  We can
    also construct it directly, which allows for other vector types:

    >>> qshift = cx.SphericalPos(r=u.Quantity(1.0, "km"),
    ...                          theta=u.Quantity(jnp.pi/2, "rad"),
    ...                          phi=u.Quantity(0, "rad"))
    >>> op = cx.ops.Translate(u.Quantity(1.0, "Gyr"), qshift)
    >>> op
    Translate(
      delta_t=Quantity(weak_f32[], unit='Gyr'),
      delta_q=SphericalPos( ... )
    )

    Translation operators can be applied to `coordinax.vecs.AbstractPos` subclasses and
    `unxt.Quantity`:

    >>> q = cx.CartesianPos3D.from_([0, 0, 0], "km")
    >>> t = u.Quantity(0, "Gyr")
    >>> newt, newq = op(t, q)
    >>> newq.x
    Quantity(Array(1., dtype=float32, ...), unit='km')
    >>> newt
    Quantity(Array(1., dtype=float32, ...), unit='Gyr')

    """

    delta_t: u.Quantity["time"] = eqx.field(converter=u.Quantity.from_)
    """The time translation.

    This parameter uses :meth:`unxt.Quantity.from_` to enable a variety of more
    convenient input types. See `unxt.Quantity` for details.
    """

    delta_q: AbstractPos = eqx.field(converter=Unless(AbstractPos, api.vector))
    """The spatial translation.

    This parameters uses :meth:`coordinax.vecs.CartesianPos3D.from_` to enable a
    variety of more convenient input types. See `coordinax.vecs.CartesianPos3D`
    for details.
    """

    # -------------------------------------------

    @property
    def is_inertial(self) -> Literal[True]:
        """Translation is an inertial-frame preserving transformation.

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
        delta_q = cast("AbstractPos3D", -self.delta_q)
        return Translate(-self.delta_t, delta_q)

    # -------------------------------------------

    @AbstractOperator.__call__.dispatch  # type: ignore[misc]
    def __call__(
        self: "Translate",
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
        return t + self.delta_t, x + self.delta_q


# -------------------------------------------


@dispatch
def simplify_op(op: Translate, /, **kwargs: Any) -> Translate | Identity:
    """Simplify a Galilean translation operator.

    Examples
    --------
    >>> import coordinax.ops as cxo

    An operator with real effect cannot be simplified:

    >>> op = cxo.Translate.from_([1, 0, 0], "m")
    >>> cxo.simplify_op(op)
    Translate(...)

    An operator with no effect can be simplified:

    >>> op = cxo.Translate.from_([0, 0, 0], "m")
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
