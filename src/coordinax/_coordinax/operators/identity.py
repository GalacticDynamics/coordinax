"""Base classes for operators on coordinates and potentials."""

__all__ = ["IdentityOperator"]

from typing import Literal, final

from jaxtyping import Shaped

from unxt import Quantity

from .base import AbstractOperator, op_call_dispatch
from coordinax._coordinax.base_pos import AbstractPosition


@final
class IdentityOperator(AbstractOperator):
    """Identity operation.

    This is the identity operation, which does nothing to the input.

    Examples
    --------
    We will work through many of the registered call signatures for the
    `IdentityOperator` class. Note that more call signatures may be registered.

    First, we make an instance of the operator:

    >>> from unxt import Quantity
    >>> import coordinax as cx

    >>> op = cx.operators.IdentityOperator()
    >>> op
    IdentityOperator()

    And the common objects we will use:

    >>> q = Quantity([1, 2, 3], "kpc")
    >>> vec = cx.CartesianPosition3D.constructor(q)

    The first call signature is for the case where the input is a vector:

    >>> op(vec) is vec
    True

    The second call signature is for a Quantity:

    >>> op(q) is q
    True

    Actually, the `Identity` operator works for any vector or quantity:

    - 1D:

    >>> q = Quantity([1], "kpc")
    >>> vec = cx.CartesianPosition1D.constructor(q)
    >>> op(vec) is vec and op(q) is q
    True

    - 2D:

    >>> q = Quantity([1, 2], "kpc")
    >>> vec = cx.CartesianPosition2D.constructor(q)
    >>> op(vec) is vec and op(q) is q
    True

    - 3D (not using a `~coordinax.CartesianPosition3D` instance):

    >>> q = Quantity([1, 2, 3], "kpc")
    >>> vec = cx.CartesianPosition3D.constructor(q).represent_as(cx.SphericalPosition)
    >>> op(vec) is vec and op(q) is q
    True

    - 4D:

    >>> q = Quantity([1, 2, 3, 4], "kpc")  # 0th elt is ct
    >>> vec4 = cx.FourVector.constructor(q)
    >>> op(vec4) is vec4 and op(q) is q
    True

    Lastly, many operators are time dependent and support a time argument. The`
    `Identity` operator will also pass through the time argument:

    >>> t = Quantity(0, "Gyr")
    >>> op(vec, t) == (vec, t)
    True

    >>> q = Quantity([1, 2, 3], "kpc")
    >>> op(q, t) == (q, t)
    True

    """

    @property
    def is_inertial(self) -> Literal[True]:
        """Identity operation is an inertial-frame preserving transform.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> from coordinax.operators import IdentityOperator

        >>> q = cx.CartesianPosition3D.constructor([1, 2, 3], "kpc")
        >>> t = Quantity(0, "Gyr")
        >>> op = IdentityOperator()
        >>> op.is_inertial
        True

        """
        return True

    @property
    def inverse(self) -> "IdentityOperator":
        """The inverse of the operator.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax.operators import IdentityOperator

        >>> op = IdentityOperator()
        >>> op.inverse
        IdentityOperator()
        >>> op.inverse is op
        True

        """
        return self

    # -------------------------------------------
    # Dispatched call signatures
    # More call signatures are registered in the `coordinax._d<X>.operate` modules.

    @op_call_dispatch(precedence=1)
    def __call__(self: "IdentityOperator", x: AbstractPosition, /) -> AbstractPosition:
        """Apply the Identity operation.

        This is the identity operation, which does nothing to the input.

        Examples
        --------
        >>> from unxt import Quantity
        >>> import coordinax as cx
        >>> from coordinax.operators import IdentityOperator

        >>> q = cx.CartesianPosition3D.constructor([1, 2, 3], "kpc")
        >>> op = IdentityOperator()
        >>> op(q) is q
        True

        """
        return x

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "IdentityOperator", x: Shaped[Quantity, "*shape"], /
    ) -> Shaped[Quantity, "*shape"]:
        """Apply the Identity operation.

        This is the identity operation, which does nothing to the input.

        Examples
        --------
        >>> from unxt import Quantity
        >>> from coordinax.operators import IdentityOperator

        >>> q = Quantity([1, 2, 3], "kpc")
        >>> op = IdentityOperator()
        >>> op(q) is q
        True

        """
        return x

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "IdentityOperator", x: AbstractPosition, t: Quantity["time"], /
    ) -> tuple[AbstractPosition, Quantity["time"]]:
        """Apply the Identity operation."""  # TODO: docstring
        return x, t

    @op_call_dispatch(precedence=1)
    def __call__(
        self: "IdentityOperator", x: Shaped[Quantity, "*shape"], t: Quantity["time"], /
    ) -> tuple[Shaped[Quantity, "*shape"], Quantity["time"]]:
        """Apply the Identity operation."""  # TODO: docstring
        return x, t
