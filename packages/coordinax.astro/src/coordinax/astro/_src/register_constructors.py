"""Register distance constructors."""

from typing import Any

import quaxed.numpy as jnp
import unxt as u

from .distance_modulus import DistanceModulus
from .parallax import Parallax, parallax_base_length


@DistanceModulus.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[DistanceModulus], p: Parallax, /, **kw: Any) -> DistanceModulus:
    """Compute distance modulus from parallax.

    >>> from coordinax.astro import DistanceModulus, Parallax
    >>> p = Parallax(1, "mas")
    >>> DistanceModulus.from_(p)
    DistanceModulus(10., 'mag')

    """
    d = parallax_base_length / jnp.tan(p)  # [AU]
    dm = 5 * jnp.log10(d.ustrip("pc")) - 5
    return cls(jnp.asarray(dm, **kw), "mag")


@Parallax.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Parallax], dm: DistanceModulus, /, **kw: Any) -> Parallax:
    """Convert distance modulus to parallax.

    >>> import unxt as u
    >>> from coordinax.astro import DistanceModulus, Parallax
    >>> dm = DistanceModulus(10, "mag")
    >>> Parallax.from_(dm).uconvert("mas").round(2)
    Parallax(1., 'mas')

    """
    d = u.Q(10 ** (1 + dm.ustrip("mag") / 5), "pc")
    p = jnp.atan2(parallax_base_length, d)
    unit = u.unit_of(p)
    return cls(jnp.asarray(p.ustrip(unit), **kw), unit)
