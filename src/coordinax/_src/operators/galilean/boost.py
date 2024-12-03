# ruff: noqa: ERA001
"""Galilean coordinate transformations."""

__all__ = ["GalileanBoostOperator"]


from dataclasses import replace
from typing import Any, Literal, final

import equinox as eqx
import jax.numpy as jnp
from plum import convert, dispatch

import quaxed.numpy as jnp
import unxt as u

from .base import AbstractGalileanOperator
from coordinax._src.operators.base import AbstractOperator
from coordinax._src.operators.identity import IdentityOperator
from coordinax._src.vectors.d3 import AbstractPos3D, CartesianVel3D
from coordinax._src.vectors.d4 import FourVector


@final
class GalileanBoostOperator(AbstractGalileanOperator):
    r"""Operator for Galilean boosts.

    The coordinate transform is given by:

    .. math::

        (t,\mathbf{x}) \mapsto (t, \mathbf{x} + \mathbf{v} t)

    where :math:`\mathbf{v}` is the boost velocity.

    Parameters
    ----------
    velocity : :class:`vector.CartesianVel3D`
        The boost velocity. This parameters uses
        :meth:`vector.CartesianVel3D.from_` to enable a variety of more
        convenient input types. See :class:`vector.CartesianVel3D` for details.

    Examples
    --------
    >>> import coordinax.operators as cxo

    >>> op = cxo.GalileanBoostOperator.from_([1.0, 2.0, 3.0], "m/s")
    >>> op
    GalileanBoostOperator( velocity=CartesianVel3D( ... ) )

    Note that the velocity is a :class:`vector.CartesianVel3D`, which was
    constructed from an array, using :meth:`vector.CartesianVel3D.from_`.

    """

    velocity: CartesianVel3D = eqx.field(converter=CartesianVel3D.from_)
    """The boost velocity.

    This parameters uses :meth:`vector.CartesianVel3D.from_` to enable a variety
    of more convenient input types. See :class:`vector.CartesianVel3D` for
    details.
    """

    # -----------------------------------------------------

    @property
    def is_inertial(self) -> Literal[True]:
        """Galilean boost is an inertial-frame preserving transform.

        Examples
        --------
        >>> import coordinax.operators as cxo

        >>> op = cxo.GalileanBoostOperator.from_([1, 2, 3], "m/s")
        >>> op.is_inertial
        True

        """
        return True

    @property
    def inverse(self) -> "GalileanBoostOperator":
        """The inverse of the operator.

        Examples
        --------
        >>> import coordinax.operators as cxo

        >>> op = cxo.GalileanBoostOperator.from_([1, 2, 3], "m/s")
        >>> op.inverse
        GalileanBoostOperator( velocity=CartesianVel3D( ... ) )

        >>> print(op.inverse.velocity)
        <CartesianVel3D (d_x[m / s], d_y[m / s], d_z[m / s])
            [-1. -2. -3.]>

        """
        return GalileanBoostOperator(-self.velocity)

    # -----------------------------------------------------

    @AbstractOperator.__call__.dispatch(precedence=1)
    def __call__(
        self: "GalileanBoostOperator", q: AbstractPos3D, t: u.Quantity["time"], /
    ) -> tuple[AbstractPos3D, u.Quantity["time"]]:
        """Apply the boost to the coordinates.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> op = cx.operators.GalileanBoostOperator.from_([1, 2, 3], "m/s")

        >>> q = cx.CartesianPos3D.from_([0, 0, 0], "m")
        >>> t = u.Quantity(1, "s")
        >>> newq, newt = op(q, t)
        >>> newt
        Quantity['time'](Array(1, dtype=int32, ...), unit='s')
        >>> print(newq)
        <CartesianPos3D (x[m], y[m], z[m])
            [1. 2. 3.]>

        """
        return q + self.velocity * t, t

    @AbstractOperator.__call__.dispatch
    def __call__(self: "GalileanBoostOperator", v4: FourVector, /) -> FourVector:
        """Apply the boost to the coordinates."""  # TODO: add example
        return replace(v4, q=v4.q + self.velocity * v4.t)


@dispatch  # type: ignore[misc]
def simplify_op(op: GalileanBoostOperator, /, **kwargs: Any) -> AbstractOperator:
    """Simplify a boost operator.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.operators as cxo

    An operator with real effect cannot be simplified:

    >>> op = cxo.GalileanBoostOperator.from_([1, 0, 0], "m/s")
    >>> cxo.simplify_op(op)
    GalileanBoostOperator(
      velocity=CartesianVel3D( ... )
    )

    An operator with no effect can be simplified:

    >>> op = cxo.GalileanBoostOperator.from_([0, 0, 0], "m/s")
    >>> cxo.simplify_op(op)
    IdentityOperator()

    """
    # Check if the velocity is zero.
    if jnp.allclose(convert(op.velocity, u.Quantity).value, jnp.zeros((3,)), **kwargs):
        return IdentityOperator()
    return op
