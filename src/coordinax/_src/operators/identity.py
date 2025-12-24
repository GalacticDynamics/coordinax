"""Identity operator."""

__all__ = ("Identity",)

from typing import Any, final

import plum

import coordinax._src.roles as cxr
from .base import AbstractOperator
from .utils import _require_role_for_pdict


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
    >>> q = u.Q([1, 2, 3], "km")
    >>> op(q) is q
    True

    We apply it to a :class:`coordinax.Cart3D` instance:

    >>> vec = cx.Vector.from_(q)
    >>> op(vec) is vec
    True

    Actually, the `Identity` operator works for any vector or quantity:

    - 1D:

    >>> q = u.Q([1], "km")
    >>> vec = cx.vecs.CartesianPos1D.from_(q)
    >>> op(vec) is vec and op(q) is q
    True

    - 2D:

    >>> q = u.Q([1, 2], "km")
    >>> vec = cx.vecs.Cart2D.from_(q)
    >>> op(vec) is vec and op(q) is q
    True

    - 3D: (not using a `~coordinax.Cart3D` instance):

    >>> q = u.Q([1, 2, 3], "km")
    >>> vec = cx.Vector.from_(q).vconvert(cx.charts.sph3d)
    >>> op(vec) is vec and op(q) is q
    True

    - 4D:

    >>> q = u.Q([1, 2, 3, 4], "km")  # 0th elt is ct
    >>> vec4 = cx.FourVector.from_(q)
    >>> op(vec4) is vec4 and op(q) is q
    True

    Lastly, many operators are time dependent and support a time argument. The
    `Identity` operator will also work with a time argument:

    >>> tau = u.Q(0, "Gyr")
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

        >>> q = u.Q([1, 2, 3], "km")
        >>> cxo.operate(cxo.Identity, {}, q) is q
        True

        >>> vec = cxo.Cart3D.from_([1, 2, 3], "km")
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


@plum.dispatch
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


# ===================================================================
# apply_op dispatches


@plum.dispatch
def apply_op(
    op: Identity,
    tau: Any,
    x: dict,  # type: ignore[type-arg]
    /,
    *,
    role: cxr.AbstractRole | None = None,
    at: Any = None,
) -> dict:  # type: ignore[type-arg]
    """Identity operator on CsDict - returns input unchanged."""
    _require_role_for_pdict(role)
    del tau, role, at  # unused
    return x
