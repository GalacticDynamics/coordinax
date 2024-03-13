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
from coordinax._d3.base import Abstract3DVector
from coordinax._d3.builtin import CartesianDifferential3D
from coordinax._d4.spacetime import FourVector
from coordinax.operators._base import AbstractOperator, op_call_dispatch
from coordinax.operators._funcs import simplify_op
from coordinax.operators._identity import IdentityOperator


@final
class GalileanBoostOperator(AbstractGalileanOperator):
    r"""Operator for Galilean boosts.

    The coordinate transform is given by:

    .. math::

        (t,\mathbf{x}) \mapsto (t, \mathbf{x} + \mathbf{v} t)

    where :math:`\mathbf{v}` is the boost velocity.

    Parameters
    ----------
    velocity : :class:`vector.CartesianDifferential3D`
        The boost velocity. This parameters uses
        :meth:`vector.CartesianDifferential3D.constructor` to enable a variety
        of more convenient input types. See
        :class:`vector.CartesianDifferential3D` for details.

    Examples
    --------
    We start with the required imports:

    >>> import quaxed.array_api as xp
    >>> from unxt import Quantity
    >>> from coordinax import CartesianDifferential3D, Cartesian3DVector
    >>> import coordinax.operators as co

    We can then create a boost operator:

    >>> op = co.GalileanBoostOperator(Quantity([1.0, 2.0, 3.0], "m/s"))
    >>> op
    GalileanBoostOperator( velocity=CartesianDifferential3D( ... ) )

    Note that the velocity is a :class:`vector.CartesianDifferential3D`, which
    was constructed from a 1D array, using
    :meth:`vector.CartesianDifferential3D.constructor`. We can also construct it
    directly:

    >>> boost = CartesianDifferential3D(d_x=Quantity(1, "m/s"), d_y=Quantity(2, "m/s"),
    ...                                 d_z=Quantity(3, "m/s"))
    >>> op = co.GalileanBoostOperator(boost)
    >>> op
    GalileanBoostOperator( velocity=CartesianDifferential3D( ... ) )

    """

    velocity: CartesianDifferential3D = eqx.field(
        converter=CartesianDifferential3D.constructor
    )
    """The boost velocity.

    This parameters uses :meth:`vector.CartesianDifferential3D.constructor` to
    enable a variety of more convenient input types. See
    :class:`vector.CartesianDifferential3D` for details.
    """

    # -----------------------------------------------------

    @property
    def is_inertial(self) -> Literal[True]:
        """Galilean boost is an inertial-frame preserving transform.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import Cartesian3DVector, CartesianDifferential3D
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
        >>> from coordinax import Cartesian3DVector, CartesianDifferential3D
        >>> from coordinax.operators import GalileanBoostOperator

        >>> op = GalileanBoostOperator(Quantity([1, 2, 3], "m/s"))
        >>> op.inverse
        GalileanBoostOperator( velocity=CartesianDifferential3D( ... ) )

        >>> op.inverse.velocity.d_x
        Quantity['speed'](Array(-1., dtype=float32), unit='m / s')

        """
        return GalileanBoostOperator(-self.velocity)

    # -----------------------------------------------------

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "GalileanBoostOperator", q: Abstract3DVector, t: Quantity["time"], /
    ) -> tuple[Abstract3DVector, Quantity["time"]]:
        """Apply the boost to the coordinates.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax import Cartesian3DVector, CartesianDifferential3D
        >>> from coordinax.operators import GalileanBoostOperator

        >>> op = GalileanBoostOperator(Quantity([1, 2, 3], "m/s"))

        >>> q = Cartesian3DVector.constructor(Quantity([0, 0, 0], "m"))
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
