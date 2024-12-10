"""Base classes for operators on coordinates and potentials."""

__all__ = ["Identity"]

from typing import Any, Literal, final

from .base import AbstractOperator


@final
class Identity(AbstractOperator):
    """Identity operation.

    This is the identity operation, which does nothing to the input.

    Examples
    --------
    We will work through many of the registered call signatures for the
    `Identity` class. Note that more call signatures may be registered.

    First, we make an instance of the operator:

    >>> import unxt as u
    >>> import coordinax as cx

    >>> op = cx.ops.Identity()
    >>> op
    Identity()

    And the common objects we will use:

    >>> q = u.Quantity([1, 2, 3], "km")
    >>> vec = cx.CartesianPos3D.from_(q)

    The first call signature is for the case where the input is a vector:

    >>> op(vec) is vec
    True

    The second call signature is for a Quantity:

    >>> op(q) is q
    True

    Actually, the `Identity` operator works for any vector or quantity:

    - 1D:

    >>> q = u.Quantity([1], "km")
    >>> vec = cx.vecs.CartesianPos1D.from_(q)
    >>> op(vec) is vec and op(q) is q
    True

    - 2D:

    >>> q = u.Quantity([1, 2], "km")
    >>> vec = cx.vecs.CartesianPos2D.from_(q)
    >>> op(vec) is vec and op(q) is q
    True

    - 3D (not using a `~coordinax.CartesianPos3D` instance):

    >>> q = u.Quantity([1, 2, 3], "km")
    >>> vec = cx.CartesianPos3D.from_(q).vconvert(cx.SphericalPos)
    >>> op(vec) is vec and op(q) is q
    True

    - 4D:

    >>> q = u.Quantity([1, 2, 3, 4], "km")  # 0th elt is ct
    >>> vec4 = cx.FourVector.from_(q)
    >>> op(vec4) is vec4 and op(q) is q
    True

    Lastly, many operators are time dependent and support a time argument. The`
    `Identity` operator will also pass through the time argument:

    >>> t = u.Quantity(0, "Gyr")
    >>> op(vec, t) == (vec, t)
    True

    >>> q = u.Quantity([1, 2, 3], "km")
    >>> op(q, t) == (q, t)
    True

    """

    @property
    def is_inertial(self) -> Literal[True]:
        """Identity operation is an inertial-frame preserving transform.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> op = cx.ops.Identity()
        >>> op.is_inertial
        True

        """
        return True

    @property
    def inverse(self) -> "Identity":
        """The inverse of the operator.

        Examples
        --------
        >>> import coordinax as cx

        >>> op = cx.ops.Identity()
        >>> op.inverse
        Identity()
        >>> op.inverse is op
        True

        """
        return self

    # -------------------------------------------
    # Dispatched call signatures
    # More call signatures are registered in the `coordinax._d<X>.operate` modules.

    @AbstractOperator.__call__.dispatch(precedence=1)
    def __call__(self: "Identity", arg: Any, /, **__: Any) -> Any:
        """Apply the Identity operation.

        This is the identity operation, which does nothing to the input.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> op = cx.ops.Identity()

        >>> q = u.Quantity([1, 2, 3], "km")
        >>> op(q) is q
        True

        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
        >>> op(vec) is vec
        True

        """
        return arg

    @AbstractOperator.__call__.dispatch(precedence=1)
    def __call__(self: "Identity", *args: Any, **__: Any) -> tuple[Any, ...]:
        """Apply the Identity operation.

        This is the identity operation, which does nothing to the input.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax as cx

        >>> op = cx.ops.Identity()

        >>> q = u.Quantity([1, 2, 3], "km")
        >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
        >>> t = u.Quantity(10, "Gyr")

        >>> op(q, t) == (q, t)
        True

        >>> op(vec, t) == (vec, t)
        True

        """
        return args
