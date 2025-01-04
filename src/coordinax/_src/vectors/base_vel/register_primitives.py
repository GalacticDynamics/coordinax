"""Register primitives for AbstractVel."""

__all__: list[str] = []


import jax
from quax import register

import unxt as u
from dataclassish import field_items

from .core import AbstractVel
from coordinax._src.vectors.base_pos import AbstractPos

# ---------------------------------------------------------


@register(jax.lax.mul_p)  # type: ignore[misc]
def _mul_vel_q(self: AbstractVel, other: u.Quantity["time"]) -> AbstractPos:
    """Multiply the vector by a time :class:`unxt.Quantity` to get a position.

    Examples
    --------
    >>> import quaxed.lax as qlax
    >>> import unxt as u
    >>> import coordinax as cx

    >>> dr = cx.vecs.RadialVel(u.Quantity(1, "m/s"))
    >>> vec = dr * u.Quantity(2, "s")
    >>> print(vec)
    <RadialPos (r[m])
        [2]>

    >>> print(qlax.mul(dr, u.Quantity(2, "s")))
    <RadialPos (r[m])
        [2]>

    """
    return self.integral_cls.from_({k[2:]: v * other for k, v in field_items(self)})
