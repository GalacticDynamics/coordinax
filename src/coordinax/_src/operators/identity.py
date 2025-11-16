"""Identity operator."""

__all__ = ("Identity",)

from typing import Any, final

from plum import dispatch

from .base import AbstractOperator


@final
class Identity(AbstractOperator):
    """Identity operation.

    This is the identity operation, which does nothing to the input.

    Examples
    --------
    >>> import coordinax as cx

    >>> op = cx.ops.Identity()
    >>> op
    Identity()

    For example, applying the `Identity` operator to a `unxt.Quantity`:

    >>> import unxt as u
    >>> q = u.Quantity([1, 2, 3], "km")
    >>> op(q) is q
    True

    We apply it to a :class:`coordinax.CartesianPos3D` instance:

    >>> vec = cx.CartesianPos3D.from_(q)
    >>> op(vec) is vec
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

    - 3D: (not using a `~coordinax.CartesianPos3D` instance):

    >>> q = u.Quantity([1, 2, 3], "km")
    >>> vec = cx.CartesianPos3D.from_(q).vconvert(cx.SphericalPos)
    >>> op(vec) is vec and op(q) is q
    True

    - 4D:

    >>> q = u.Quantity([1, 2, 3, 4], "km")  # 0th elt is ct
    >>> vec4 = cx.FourVector.from_(q)
    >>> op(vec4) is vec4 and op(q) is q
    True

    Lastly, many operators are time dependent and support a time argument. The
    `Identity` operator will also work with a time argument:

    >>> tau = u.Quantity(0, "Gyr")
    >>> op(tau, vec)

    >>> op(tau, q)

    """

    @classmethod
    def operate(cls, _: dict[str, Any], arg: Any, /, **__: Any) -> Any:
        """Apply the :class:`coordinax.ops.Identity` operation.

        This is the identity operation, which does nothing to the input.

        Examples
        --------
        >>> import unxt as u
        >>> import coordinax.ops as cxo

        >>> q = u.Quantity([1, 2, 3], "km")
        >>> cxo.operate(cxo.Identity, {}, q) is q
        True

        >>> vec = cxo.CartesianPos3D.from_([1, 2, 3], "km")
        >>> cxo.operate(cxo.Identity, {}, vec) is vec
        True

        """
        return arg

    @property
    def inverse(self) -> "Identity":
        """The inverse of the operator is the operator itself.

        Examples
        --------
        >>> import coordinax as cx

        >>> op = cx.ops.Identity()
        >>> op.inverse is op
        True

        """
        return self


# ===================================================================


@dispatch
def simplify(op: Identity, /, **__: Any) -> Identity:
    """Simplify a :class:`coordinax.ops.Identity` operator.

    Examples
    --------
    The :class:`coordinax.ops.Identity` operator is the simplest operator and
    cannot be simplified further:

    >>> import coordinax.ops as cxo

    >>> op = cxo.Identity()
    >>> simplified = cxo.simplify(op)
    >>> simplified == op
    True

    """
    return op
