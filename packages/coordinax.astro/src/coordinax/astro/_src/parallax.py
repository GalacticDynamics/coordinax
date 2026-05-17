"""Parallax distance quantity."""

__all__ = ("Parallax",)

from dataclasses import KW_ONLY

from jaxtyping import Array, ArrayLike, Shaped
from typing import Any, final

import equinox as eqx
import jax.numpy as jnp

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import BareQuantity

import coordinax.distances as cxd
from .constants import ANGLE

parallax_base_length = u.Q(jnp.array(1), "AU")


@final
class Parallax(cxd.AbstractDistance):
    """Parallax distance quantity.

    Examples
    --------
    >>> from coordinax.astro import Parallax
    >>> Parallax(1, "mas")
    Parallax(1, 'mas')

    The units are checked to have angle dimensions.

    >>> try: Parallax(1, "pc")
    ... except ValueError as e: print(e)
    Parallax must have angular dimensions.

    The parallax is checked to be non-negative by default.

    >>> try: Parallax(-1, "mas")
    ... except Exception: print("negative")
    negative

    To disable this check, set `check_negative=False`.

    >>> Parallax(-1, "mas", check_negative=False)
    Parallax(-1, 'mas', check_negative=False)

    """

    value: Shaped[Array, "*shape"] = eqx.field(
        converter=u.quantity.convert_to_quantity_value
    )
    """The value of the `unxt.AbstractQuantity`."""

    unit: u.AbstractUnit = eqx.field(static=True, converter=u.unit)  # ty: ignore[invalid-assignment]
    """The unit associated with this value."""

    _: KW_ONLY
    check_negative: bool = eqx.field(default=True, static=True, compare=False)
    """Whether to check that the parallax is strictly non-negative.

    Theoretically the parallax must be strictly non-negative ($\tan(p) = 1
    AU / d$), however noisy direct measurements of the parallax can be negative.
    """

    def __check_init__(self) -> None:
        """Check the initialization."""
        if u.dimension_of(self) != ANGLE:
            msg = "Parallax must have angular dimensions."
            raise ValueError(msg)

        if self.check_negative:  # pylint: disable=unreachable
            eqx.error_if(
                self.value,
                jnp.any(jnp.less(self.value, 0)),
                "Parallax must be non-negative.",
            )


@Parallax.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Parallax], value: ArrayLike, unit: Any, /, **kw: Any) -> Parallax:
    """Construct a distance.

    >>> import unxt as u
    >>> from coordinax.astro import Parallax

    >>> Parallax.from_(1, "mas")
    Parallax(1, 'mas')

    """
    return cls(jnp.asarray(value, **kw), unit)


@Parallax.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Parallax], p: Parallax, /, **kw: Any) -> Parallax:
    """Compute parallax from parallax.

    >>> import unxt as u
    >>> from coordinax.astro import Parallax

    >>> p = Parallax(1, "mas")
    >>> Parallax.from_(p) is p
    True

    >>> Parallax.from_(p, dtype=float)
    Parallax(1., 'mas')

    """
    if len(kw) == 0:
        return p
    return jnp.asarray(p, **kw)  # ty: ignore[invalid-return-type]


@Parallax.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Parallax], p: u.Q["angle"], /, **kw: Any) -> Parallax:
    """Compute parallax from parallax.

    >>> import unxt as u
    >>> from coordinax.astro import Parallax

    >>> q = u.Q(1, "mas")
    >>> Parallax.from_(q, dtype=float)
    Parallax(1., 'mas')

    """
    unit = u.unit_of(p)
    return cls(jnp.asarray(p.ustrip(unit), **kw), unit)


@Parallax.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Parallax],
    d: cxd.Distance | u.Q["length"],
    /,
    **kw: Any,
) -> Parallax:
    """Compute parallax from distance.

    >>> import unxt as u
    >>> from coordinax.astro import Parallax

    >>> d = cxd.Distance(10, "pc")
    >>> Parallax.from_(d).uconvert("mas").round(2)
    Parallax(100., 'mas')

    >>> q = u.Q(10, "pc")
    >>> Parallax.from_(q).uconvert("mas").round(2)
    Parallax(100., 'mas')

    """
    p = jnp.atan2(parallax_base_length, d)
    return cls(jnp.asarray(p.value, **kw), p.unit)  # ty: ignore[unresolved-attribute]


@Parallax.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Parallax], dm: u.Q["mag"], /, **kw: Any) -> Parallax:
    """Convert distance modulus to parallax.

    >>> import unxt as u
    >>> from coordinax.astro import Parallax

    >>> dm = u.Q(10, "mag")
    >>> Parallax.from_(dm).uconvert("mas").round(2)
    Parallax(1., 'mas')

    """
    d = BareQuantity(10 ** (1 + dm.ustrip("mag") / 5), "pc")
    p = jnp.atan2(parallax_base_length, d)
    unit = u.unit_of(p)
    return cls(jnp.asarray(p.ustrip(unit), **kw), unit)  # ty: ignore[unresolved-attribute]


@cxd.Distance.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[cxd.Distance], p: Parallax, /, **kw: Any) -> cxd.Distance:
    """Compute distance from parallax.

    >>> import coordinax.distances as cxd
    >>> from coordinax.astro import Parallax

    >>> p = Parallax(1, "mas")
    >>> cxd.Distance.from_(p).uconvert("pc").round(2)
    Distance(1000., 'pc')

    """
    d = parallax_base_length / jnp.tan(p)  # [AU]
    unit = u.unit_of(d)
    return cls(jnp.asarray(d.ustrip(unit), **kw), unit)
