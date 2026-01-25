"""Identity operator."""

__all__ = ("Identity",)

from typing import Any, final

import plum
from jax.typing import ArrayLike

import unxt as u

from .base import AbstractOperator
from coordinax._src import charts as cxc, roles as cxr
from coordinax._src.custom_types import CsDict


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
    >>> vec = cx.Vector.from_(q)
    >>> op(vec) is vec and op(q) is q
    True

    - 2D:

    >>> q = u.Q([1, 2], "km")
    >>> vec = cx.Vector.from_(q)
    >>> op(vec) is vec and op(q) is q
    True

    - 3D: (using a spherical chart):

    >>> q = u.Q([1, 2, 3], "km")
    >>> vec = cx.Vector.from_(q).vconvert(cx.charts.sph3d)
    >>> op(vec) is vec and op(q) is q
    True

    Lastly, many operators are time dependent and support a time argument. The
    `Identity` operator will also work with a time argument:

    >>> tau = u.Q(0, "Gyr")
    >>> op(tau, vec)  # doctest: +NORMALIZE_WHITESPACE
    Vector(
      data={
        'r': Q(3.74165739, 'km'), 'theta': Q(0.64052231, 'rad'),
        'phi': Q(1.10714872, 'rad')
      },
      chart=Spherical3D(),
      role=Point()
    )

    >>> op(tau, q)
    Quantity(Array([1, 2, 3], dtype=int64), unit='km')

    """

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
    role: None,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: CsDict | u.AbstractQuantity | ArrayLike,
    /,
) -> CsDict | u.AbstractQuantity | ArrayLike:
    """Identity operator (role=None) - returns input unchanged."""
    del op, tau, role, chart  # unused
    return x


@plum.dispatch
def apply_op(
    op: Identity,
    tau: Any,
    role: cxr.AbstractRole,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: u.AbstractQuantity,
    /,
) -> u.AbstractQuantity:
    """Identity operator on Quantity - returns input unchanged."""
    del op, tau, role, chart  # unused
    return x


@plum.dispatch
def apply_op(
    op: Identity,
    tau: Any,
    role: cxr.AbstractRole,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: CsDict,
    /,
) -> CsDict:
    """Identity operator on CsDict - returns input unchanged."""
    del op, tau, role, chart  # unused
    return x


@plum.dispatch
def apply_op(
    op: Identity,
    tau: Any,
    role: cxr.AbstractRole,
    chart: cxc.AbstractChart,  # type: ignore[type-arg]
    x: ArrayLike,
    /,
) -> ArrayLike:
    """Identity operator on ArrayLike - returns input unchanged."""
    del op, tau, role, chart  # unused
    return x
