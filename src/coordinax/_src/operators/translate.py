"""Coordinate translations."""

__all__ = ["Translate"]


from typing import Any, Literal, final

import equinox as eqx
from plum import convert, dispatch

import quaxed.numpy as jnp
import unxt as u
from dataclassish.converters import Unless

from coordinax._src.operators.base import AbstractOperator
from coordinax._src.operators.identity import Identity
from coordinax._src.vectors import api
from coordinax._src.vectors.base_pos import AbstractPos


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

    >>> op = cx.ops.Translate.from_([1.0, 2.0, 3.0], "km")
    >>> op
    Translate(
      delta_t=Quantity(f32[], unit='s'),
      delta_q=CartesianPos3D( ... )
    )
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

    delta: AbstractPos = eqx.field(converter=Unless(AbstractPos, api.vector))
    """The translation amount.

    This supports both position translations, by passing in an `AbstractPos`
    subclass or a `Quantity` with length unit, and time translations, by passing in a
    `u.Quantity` with a time unit. This also supports a simultaneous translation in
    position and time by specifying a `FourVector`
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
        return Translate(-self.delta)

    # -------------------------------------------

    @AbstractOperator.__call__.dispatch  # type: ignore[misc]
    def __call__(
        self: "Translate",
        x: AbstractPos,
        /,
        **__: Any,
    ) -> AbstractPos:
        """Apply the translation to the coordinates.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx
        >>> import coordinax.ops as cxo

        For positions only:

        >>> dq = cx.CartesianPos3D.from_([1, 1, 1], "km")
        >>> op = cx.ops.Translate(dq)
        >>> q = cx.CartesianPos3D.from_([1, 2, 3], "km")
        >>> newq = op(q)
        >>> newq.x
        Quantity(Array(2, dtype=int32), unit='km')
        >>> newq.y
        Quantity(Array(3, dtype=int32), unit='km')
        >>> newq.z
        Quantity(Array(4, dtype=int32), unit='km')

        This also works for position + time translations using a `FourVector`:

        >>> dq4 = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([4, 5, 6], "m"))
        >>> op4 = cx.ops.Translate(dq4)

        Now let's apply it to a `FourVector`:

        >>> q4 = cx.FourVector(t=u.Quantity(0, "s"), q=u.Quantity([1, 2, 3], "m"))
        >>> newq4 = op4(q4)
        >>> newq4
        FourVector( t=Quantity(...), q=CartesianPos3D( ... ) )

        >>> newq4.q.x
        Quantity(Array(5., dtype=float32), unit='km')
        >>> newq4.t
        Quantity(Array(1., dtype=float32), unit='s')

        TODO: Now on a VelocityBoost:

        >>> op = cx.ops.VelocityBoost.from_([1, 2, 3], "m/s")

        >>> v4 = cx.FourVector.from_([0, 0, 0, 0], "m")
        >>> newv4 = op(v4)
        >>> print(newv4)
        <FourVector: (t[m s / km], q=(x, y, z) [m])
            [0. 0. 0. 0.]>

        """
        return x + self.delta


@AbstractOperator.from_.dispatch  # type: ignore[misc]
def from_(cls: type[Translate], delta: u.Quantity["length"], /) -> Translate:
    """Construct from a spatial offset.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.ops as cxo

    >>> q = u.Quantity([1.0, 2.0, 3.0], "km")
    >>> op = cxo.Translation.from_(q)

    """
    return cls(delta=api.vector(delta))


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
    q = convert(op.delta, u.Quantity)
    if jnp.allclose(q.value, 0, **kwargs):
        return Identity()

    return op


@dispatch
def simplify_op(op1: Translate, op2: Translate, /) -> Translate:
    """Combine two translations into a single translation.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> qshift = cx.CartesianPos3D.from_([1, 0, 0], "km")
    >>> op1 = cx.ops.Translate(qshift)

    >>> qshift2 = cx.CartesianPos3D.from_([0, 1, 0], "km")
    >>> op2 = cx.ops.Translate(qshift2)

    >>> op3 = cx.ops.simplify_op(op1, op2)
    >>> op3
    Translate(
      delta=CartesianPos3D( ... )
    )

    """
    return Translate(op1.delta + op2.delta)
