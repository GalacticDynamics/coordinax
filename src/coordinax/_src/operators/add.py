"""transformations by coordinate addition."""

__all__ = ("Add",)


from collections.abc import Callable
from typing import Any, Union, final

import equinox as eqx
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

from .base import AbstractOperator, Neg
from .custom_types import CanAddandNeg
from .pipe import Pipe
from coordinax._src.vectors.base_pos import AbstractPos
from coordinax._src.operators.base import AbstractOperator
from coordinax._src.operators.identity import Identity


@final
class Add(AbstractOperator):
    r"""Operator for addition.

    This can be used to accomplish spatio-temporal translations of coordinates.

    Parameters
    ----------
    delta
        The thing to add.

        If it's a callable...

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax as cx

    We can then create a translation operator:

    >>> op = cx.ops.Add.from_([2.0, 3.0, 4.0], "km")
    >>> op
    Add(delta=Quantity(f32[3], unit='km'))

    Note that the translation is a `unxt.Quantity` and a
    `coordinax.vecs.AbstractPos`, which was constructed from a 1D array, using
    :meth:`coordinax.vecs.AbstractPos.from_`.  We can also construct it
    directly, which allows for other vector types.

    >>> qshift = cx.SphericalPos(r=u.Quantity(1.0, "km"),
    ...                          theta=u.Quantity(jnp.pi/2, "rad"),
    ...                          phi=u.Quantity(0, "rad"))
    >>> op = cx.ops.Add(qshift)
    >>> print(op)
    Add(<SphericalPos: (r[km], theta[rad], phi[rad])
            [1.    1.571 0.   ]>)

    Translation operators can be applied to `coordinax.vecs.AbstractPos`:

    >>> q = cx.CartesianPos3D.from_([0, 0, 0], "km")
    >>> newq = op(q)
    >>> print(newq.round(2))
    <CartesianPos3D: (x, y, z) [km]
        [ 1.  0. -0.]>

    """

    delta: CanAddandNeg | Callable[[Any], Any]
    """The translation."""

    right_add: bool = eqx.field(default=True, static=True)

    # -------------------------------------------

    @classmethod
    @dispatch
    def operate(cls, params: dict[str, Any], arg: Any, /, **_: Any) -> Any:
        """Apply the :class:`coordinax.ops.Add` operation.

        This is the identity operation, which does nothing to the input.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax.vecs as cxv
        >>> import coordinax.ops as cxo

        >>> q = u.Quantity([1, 2, 3], "km")
        >>> cxo.operate(cxo.Add, {"delta": u.Quantity([1, 1, 1], "km")}, q)
        Quantity(Array([2, 3, 4], dtype=int32), unit='km')

        >>> vec = cxv.CartesianPos3D.from_([1, 2, 3], "km")
        >>> params = {"delta": cxv.CartesianPos3D.from_([1, 1, 1], "km")}
        >>> print(cxo.operate(cxo.Add, params, vec))
        <CartesianPos3D: (x, y, z) [km]
            [2 3 4]>

        """
        return (
            arg + params["delta"]
            if params.get("right_add", True)
            else params["delta"] + arg
        )

    @property
    def inverse(self) -> "Add":
        """The inverse of the operator.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> qshift = cx.CartesianPos3D.from_([1, 1, 1], "km")
        >>> op = cx.ops.Add.from_(qshift)

        >>> print(op.inverse)
        Add(<CartesianPos3D: (x, y, z) [km]
            [-1 -1 -1]>)

        """
        delta = self.delta
        inv = -delta if (not callable(delta) or isinstance(delta, Neg)) else Neg(delta)
        return Add(inv)

    # ===============================================================
    # Python API

    def __add__(self, other: object, /) -> Union["Add", Pipe]:
        """Combine two translations into a single translation."""
        if not isinstance(other, Add):
            return NotImplemented

        if not callable(self.delta) and not callable(other.delta):
            return Add(self.delta + other.delta)
        return Pipe((self, other))

    def __neg__(self, /) -> "Add":
        """Return negative of the translation.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> qshift = cx.CartesianPos3D.from_([1, 1, 1], "km")
        >>> op = cx.ops.Add(qshift)
        >>> print(-op)
        Add(<CartesianPos3D: (x, y, z) [km]
            [-1 -1 -1]>)

        """
        return self.inverse


# ===================================================================


# Demonstrating Add fast-path.
@dispatch
def operate(op: Add, tau: Any, arg: Any, /, **_: Any) -> Any:
    """Apply the :class:`coordinax.ops.Add` operation."""
    delta = op.delta(tau) if callable(op.delta) else op.delta
    return arg + delta if op.right_add else delta + arg


# ===================================================================


@dispatch
def simplify(op: Add, /, **kwargs: Any) -> Add | Identity:
    """Simplify an `Add` operator.

    Examples
    --------
    >>> import coordinax.ops as cxo

    An operator with real effect cannot be simplified:

    >>> op = cxo.Add.from_([3e8, 1, 0, 0], "m")
    >>> cxo.simplify(op)
    Add(delta=Quantity(f32[4], unit='m'))

    An operator with no effect can be simplified:

    >>> op = cxo.Add.from_([0, 0, 0, 0], "m")
    >>> cxo.simplify(op)
    Identity()

    """
    # Check if the translation is zero. Need to strip any units.
    if jnp.allclose(u.ustrip(AllowValue, op.delta), 0, **kwargs):
        return Identity()  # type: ignore[no-untyped-call]

    return op
