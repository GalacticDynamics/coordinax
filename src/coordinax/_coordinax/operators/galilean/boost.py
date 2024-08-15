# ruff: noqa: ERA001
"""Galilean coordinate transformations."""

__all__ = ["GalileanBoostOperator"]


from dataclasses import replace
from typing import Any, Literal, final

import equinox as eqx
import jax.numpy as jnp
from plum import convert

import quaxed.array_api as xp
from unxt import Quantity

from .base import AbstractGalileanOperator
from coordinax._coordinax.d3.base import AbstractPosition3D
from coordinax._coordinax.d3.cartesian import CartesianVelocity3D
from coordinax._coordinax.d4.spacetime import FourVector
from coordinax._coordinax.operators.base import AbstractOperator, op_call_dispatch
from coordinax._coordinax.operators.funcs import simplify_op
from coordinax._coordinax.operators.identity import IdentityOperator


@final
class GalileanBoostOperator(AbstractGalileanOperator):
    r"""Operator for Galilean boosts.

    The coordinate transform is given by:

    .. math::

        (t,\mathbf{x}) \mapsto (t, \mathbf{x} + \mathbf{v} t)

    where :math:`\mathbf{v}` is the boost velocity.

    Parameters
    ----------
    velocity : :class:`vector.CartesianVelocity3D`
        The boost velocity. This parameters uses
        :meth:`vector.CartesianVelocity3D.constructor` to enable a variety
        of more convenient input types. See
        :class:`vector.CartesianVelocity3D` for details.

    Examples
    --------
    We start with the required imports:

    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import coordinax.operators as co

    We can then create a boost operator:

    >>> op = co.GalileanBoostOperator(Quantity([1.0, 2.0, 3.0], "m/s"))
    >>> op
    GalileanBoostOperator( velocity=CartesianVelocity3D( ... ) )

    Note that the velocity is a :class:`vector.CartesianVelocity3D`, which
    was constructed from a 1D array, using
    :meth:`vector.CartesianVelocity3D.constructor`. We can also construct it
    directly:

    >>> boost = cx.CartesianVelocity3D.constructor([1, 2, 3], "m/s")
    >>> op = co.GalileanBoostOperator(boost)
    >>> op
    GalileanBoostOperator( velocity=CartesianVelocity3D( ... ) )

    """

    velocity: CartesianVelocity3D = eqx.field(converter=CartesianVelocity3D.constructor)
    """The boost velocity.

    This parameters uses :meth:`vector.CartesianVelocity3D.constructor` to
    enable a variety of more convenient input types. See
    :class:`vector.CartesianVelocity3D` for details.
    """

    # -----------------------------------------------------

    @property
    def is_inertial(self) -> Literal[True]:
        """Galilean boost is an inertial-frame preserving transform.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax.operators import GalileanBoostOperator

        >>> op = GalileanBoostOperator(Quantity([1, 2, 3], "m/s"))
        >>> op.is_inertial
        True

        """
        return True

    @property
    def inverse(self) -> "GalileanBoostOperator":
        """The inverse of the operator.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax.operators import GalileanBoostOperator

        >>> op = GalileanBoostOperator(Quantity([1, 2, 3], "m/s"))
        >>> op.inverse
        GalileanBoostOperator( velocity=CartesianVelocity3D( ... ) )

        >>> op.inverse.velocity.d_x
        Quantity['speed'](Array(-1., dtype=float32), unit='m / s')

        """
        return GalileanBoostOperator(-self.velocity)

    # -----------------------------------------------------

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "GalileanBoostOperator", q: AbstractPosition3D, t: Quantity["time"], /
    ) -> tuple[AbstractPosition3D, Quantity["time"]]:
        """Apply the boost to the coordinates.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax.operators import GalileanBoostOperator

        >>> op = GalileanBoostOperator(Quantity([1, 2, 3], "m/s"))

        >>> q = cx.CartesianPosition3D.constructor([0, 0, 0], "m")
        >>> t = Quantity(1, "s")
        >>> newq, newt = op(q, t)
        >>> newt
        Quantity['time'](Array(1, dtype=int32, ...), unit='s')
        >>> newq.x
        Quantity['length'](Array(1., dtype=float32), unit='m')

        """
        return q + self.velocity * t, t

    @op_call_dispatch
    def __call__(self: "GalileanBoostOperator", v4: FourVector, /) -> FourVector:
        """Apply the boost to the coordinates."""  # TODO: add example
        return replace(v4, q=v4.q + self.velocity * v4.t)


@simplify_op.register
def _simplify_op_boost(op: GalileanBoostOperator, /, **kwargs: Any) -> AbstractOperator:
    """Simplify a boost operator."""
    # Check if the velocity is zero.
    if jnp.allclose(convert(op.velocity, Quantity).value, xp.zeros((3,)), **kwargs):
        return IdentityOperator()
    return op
