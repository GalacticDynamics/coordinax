"""Distance quantities."""

__all__ = ("Distance",)

from dataclasses import KW_ONLY

from jaxtyping import Array, ArrayLike, Shaped
from typing import Any, cast, final

import equinox as eqx
import jax.numpy as jnp

import quaxed.numpy as jnp
import unxt as u

from .base import AbstractDistance
from .constants import LENGTH

parallax_base_length = u.Q(jnp.array(1), "AU")


@final
class Distance(AbstractDistance):
    """Distance quantities.

    The distance is a quantity with dimensions of length.

    Examples
    --------
    >>> import coordinax.distances as cxd
    >>> cxd.Distance(10, "km")
    Distance(10, 'km')

    The units are checked to have length dimensions.

    >>> try: cxd.Distance(10, "s")
    ... except ValueError as e: print(e)
    Distance must have dimensions length.

    """

    value: Shaped[Array, "*shape"] = eqx.field(
        converter=u.quantity.convert_to_quantity_value
    )
    """The distance value."""

    unit: u.AbstractUnit = eqx.field(static=True, converter=u.unit)
    """The unit associated with this value."""

    _: KW_ONLY
    check_negative: bool = eqx.field(default=True, static=True, compare=False)
    """Whether to check that the distance is strictly non-negative."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        if u.dimension_of(self) != LENGTH:
            msg = "Distance must have dimensions length."
            raise ValueError(msg)

        if self.check_negative:  # pylint: disable=unreachable
            eqx.error_if(
                self.value,
                jnp.any(jnp.less(self.value, 0)),
                "Distance must be non-negative.",
            )


@Distance.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Distance], value: ArrayLike, unit: Any, /, **kw: Any) -> Distance:
    """Construct a distance.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> cxd.Distance.from_(1, "kpc")
    Distance(1, 'kpc')

    """
    return cls(jnp.asarray(value, **kw), unit)


@Distance.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Distance], d: Distance, /, **kw: Any) -> Distance:
    """Compute distance from distance.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> d = cxd.Distance(1, "kpc")
    >>> cxd.Distance.from_(d) is d
    True

    >>> cxd.Distance.from_(d, dtype=float)
    Distance(1., 'kpc')

    """
    if len(kw) == 0:
        return d
    return cast("Distance", jnp.asarray(d, **kw))


@Distance.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(cls: type[Distance], d: u.Q["length"], /, **kw: Any) -> Distance:  # type: ignore[type-arg]
    """Compute distance from distance.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> q = u.Q(1, "kpc")
    >>> cxd.Distance.from_(q, dtype=float)
    Distance(1., 'kpc')

    """
    unit = u.unit_of(d)
    return cls(jnp.asarray(d.ustrip(unit), **kw), unit)


@Distance.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Distance],
    p: u.Q["angle"],  # type: ignore[type-arg]
    /,
    **kw: Any,
) -> Distance:
    """Compute distance from parallax.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> q = u.Q(1, "mas")
    >>> cxd.Distance.from_(q).uconvert("pc").round(2)
    Distance(1000., 'pc')

    """
    d = parallax_base_length / jnp.tan(p)  # [AU]
    unit = u.unit_of(d)
    return cls(jnp.asarray(d.ustrip(unit), **kw), unit)


@Distance.from_.dispatch  # ty: ignore[unresolved-attribute]
def from_(
    cls: type[Distance],
    dm: u.Q["mag"],  # type: ignore[type-arg]
    /,
    **kw: Any,
) -> Distance:
    """Compute distance from distance modulus.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> q = u.Q(10, "mag")
    >>> cxd.Distance.from_(q).uconvert("pc").round(2)
    Distance(1000., 'pc')

    """
    d = 10 ** (1 + dm.ustrip("mag") / 5)
    return cls(jnp.asarray(d, **kw), "pc")
