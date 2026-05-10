"""Identity operator."""

__all__ = ("Identity", "identity")

from typing import Any, final

import plum

from .base import AbstractTransform
from coordinax.transforms._src.groups import DiffeomorphismGroup, IdentityGroup


@final
class Identity(AbstractTransform):
    """Identity operation.

    This is the identity operation, which does nothing to the input.

    Examples
    --------
    >>> import coordinax.main as cx
    >>> import coordinax.transforms as cxfm
    >>> import unxt as u

    >>> op = cxfm.Identity()
    >>> op
    Identity()

    For example, applying the `Identity` operator to a `unxt.Quantity`:

    >>> q = u.Q([1, 2, 3], "km")
    >>> op(q) is q
    True

    We apply it to a {class}`coordinax.Cart3D` instance:

    >>> vec = cx.Point.from_(q)
    >>> op(vec) is vec
    True

    Actually, the `Identity` operator works for any vector or quantity:

    - 1D:

    >>> q = u.Q([1], "km")
    >>> vec = cx.Point.from_(q)
    >>> op(vec) is vec and op(q) is q
    True

    - 2D:

    >>> q = u.Q([1, 2], "km")
    >>> vec = cx.Point.from_(q)
    >>> op(vec) is vec and op(q) is q
    True

    - 3D: (using a spherical chart):

    >>> q = u.Q([1, 2, 3], "km")
    >>> vec = cx.Point.from_(q).cconvert(cx.sph3d)
    >>> op(vec) is vec and op(q) is q
    True

    Lastly, many operators are time dependent and support a time argument. The
    `Identity` operator will also work with a time argument:

    >>> tau = u.Q(0, "Gyr")
    >>> op(tau, vec)
    Point(
      {'r': Q(3.74165739, 'km'), 'theta': Q(0.64052231, 'rad'), 'phi': Q(1.10714872, 'rad')},
      chart=Spherical3D(M=Rn(3)), M=Rn(3)
    )

    >>> op(tau, q)
    Q([1, 2, 3], 'km')

    """  # noqa: E501

    @classmethod
    def groups(cls) -> frozenset[type]:
        """Return the groups to which this map belongs."""
        del cls
        return frozenset((IdentityGroup, DiffeomorphismGroup))

    @property
    def inverse(self) -> "Identity":
        """The inverse of the map is the map itself.

        Examples
        --------
        >>> import coordinax.transforms as cxfm

        >>> op = cxfm.Identity()
        >>> op.inverse is op
        True

        """
        return self


identity = Identity()  # convenience instance
"""Identity operator instance."""

# ===================================================================


@plum.dispatch
def simplify(op: Identity, /, **__: Any) -> Identity:
    """Simplify a {class}`coordinax.transforms.Identity` operator.

    Examples
    --------
    The {class}`coordinax.transforms.Identity` operator is the simplest operator and
    cannot be simplified further:

    >>> import coordinax.transforms as cxfm

    >>> op = cxfm.Identity()
    >>> simplified = cxfm.simplify(op)
    >>> simplified == op
    True

    """
    return op


# ===================================================================
# `act` dispatches


# Precedence=1 because this is a catch-all for any input type.
@plum.dispatch(precedence=1)  # ty: ignore[no-matching-overload]
def act(op: Identity, tau: Any, x: Any, /, *args: Any, **kw: Any) -> Any:
    """Identity operator - returns input unchanged.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.transforms as cxfm

    >>> op = cxfm.Identity()

    >>> q = [1, 2, 3]
    >>> cxfm.act(op, None, q) is q
    True

    >>> q = u.Q([1, 2, 3], "km")
    >>> cxfm.act(op, None, q) is q
    True

    >>> data = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
    >>> cxfm.act(op, None, data) is data
    True

    >>> v = cx.Point.from_(u.Q([1, 2, 3], "m"))
    >>> cxfm.act(op, None, v) is v
    True

    """
    del args, tau, kw  # unused
    return x
