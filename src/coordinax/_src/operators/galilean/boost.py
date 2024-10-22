# ruff: noqa: ERA001
"""Galilean coordinate transformations."""

__all__ = ["GalileanBoostOperator"]


from dataclasses import replace
from typing import Any, Literal, final

import equinox as eqx
import jax.numpy as jnp
from plum import convert

import quaxed.numpy as jnp
from unxt import Quantity

from .base import AbstractGalileanOperator
from coordinax._src.d3.base import AbstractPos3D
from coordinax._src.d3.cartesian import CartesianVel3D
from coordinax._src.d4.spacetime import FourVector
from coordinax._src.operators.base import AbstractOperator, op_call_dispatch
from coordinax._src.operators.funcs import simplify_op
from coordinax._src.operators.identity import IdentityOperator


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
    We start with the required imports:

    >>> from unxt import Quantity
    >>> import coordinax as cx
    >>> import coordinax.operators as co

    We can then create a boost operator:

    >>> op = co.GalileanBoostOperator(Quantity([1.0, 2.0, 3.0], "m/s"))
    >>> op
    GalileanBoostOperator( velocity=CartesianVel3D( ... ) )

    Note that the velocity is a :class:`vector.CartesianVel3D`, which was
    constructed from a 1D array, using :meth:`vector.CartesianVel3D.from_`. We
    can also construct it directly:

    >>> boost = cx.CartesianVel3D.from_([1, 2, 3], "m/s")
    >>> op = co.GalileanBoostOperator(boost)
    >>> op
    GalileanBoostOperator( velocity=CartesianVel3D( ... ) )

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
        GalileanBoostOperator( velocity=CartesianVel3D( ... ) )

        >>> op.inverse.velocity.d_x
        Quantity['speed'](Array(-1., dtype=float32), unit='m / s')

        """
        return GalileanBoostOperator(-self.velocity)

    # -----------------------------------------------------

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "GalileanBoostOperator", q: AbstractPos3D, t: Quantity["time"], /
    ) -> tuple[AbstractPos3D, Quantity["time"]]:
        """Apply the boost to the coordinates.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax.operators import GalileanBoostOperator

        >>> op = GalileanBoostOperator(Quantity([1, 2, 3], "m/s"))

        >>> q = cx.CartesianPos3D.from_([0, 0, 0], "m")
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
    if jnp.allclose(convert(op.velocity, Quantity).value, jnp.zeros((3,)), **kwargs):
        return IdentityOperator()
    return op
