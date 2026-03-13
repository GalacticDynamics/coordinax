"""Distance quantities."""

__all__ = ("Distance", "DistanceModulus", "Parallax")

from dataclasses import KW_ONLY

from jaxtyping import Array, ArrayLike, Shaped
from typing import Any, final

import equinox as eqx
import jax.numpy as jnp

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import BareQuantity

from .base import AbstractDistance
from .constants import ANGLE, LENGTH

parallax_base_length = u.Q(jnp.array(1), "AU")
distance_modulus_base_distance = u.Q(jnp.array(10), "pc")


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


@Distance.from_.dispatch
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


@Distance.from_.dispatch
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
    return jnp.asarray(d, **kw)


@Distance.from_.dispatch
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


@Distance.from_.dispatch
def from_(
    cls: type[Distance],
    p: "Parallax",
    /,
    **kw: Any,
) -> Distance:
    """Compute distance from parallax.

    Examples
    --------
    >>> import coordinax.distances as cxd

    >>> p = cxd.Parallax(1, "mas")
    >>> cxd.Distance.from_(p).uconvert("pc").round(2)
    Distance(1000., 'pc')

    """
    d = parallax_base_length / jnp.tan(p)  # [AU]
    unit = u.unit_of(d)
    return cls(jnp.asarray(d.ustrip(unit), **kw), unit)


@Distance.from_.dispatch
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


@Distance.from_.dispatch
def from_(
    cls: type[Distance],
    dm: "DistanceModulus",  # type: ignore[type-arg]
    /,
    **kw: Any,
) -> Distance:
    """Compute distance from distance modulus.

    Examples
    --------
    >>> import coordinax.distances as cxd

    >>> dm = cxd.DistanceModulus(10, "mag")
    >>> cxd.Distance.from_(dm).uconvert("pc").round(2)
    Distance(1000., 'pc')

    """
    d = 10 ** (1 + dm.ustrip("mag") / 5)
    return cls(jnp.asarray(d, **kw), "pc")


@Distance.from_.dispatch
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


##############################################################################


@final
class DistanceModulus(AbstractDistance):
    """Distance modulus quantity.

    Examples
    --------
    >>> from coordinax.distances import DistanceModulus
    >>> DistanceModulus(10, "mag")
    DistanceModulus(10, 'mag')

    The units are checked to have magnitude dimensions.

    >>> try: DistanceModulus(10, "pc")
    ... except ValueError as e: print(e)
    Distance modulus must have units of magnitude.

    """

    value: Shaped[Array, "*shape"] = eqx.field(
        converter=u.quantity.convert_to_quantity_value
    )
    """The value of the `unxt.AbstractQuantity`."""

    unit: u.AbstractUnit = eqx.field(static=True, converter=u.unit)
    """The unit associated with this value."""

    def __check_init__(self) -> None:
        """Check the initialization."""
        if self.unit != u.unit("mag"):
            msg = "Distance modulus must have units of magnitude."
            raise ValueError(msg)


@DistanceModulus.from_.dispatch
def from_(
    cls: type[DistanceModulus], value: ArrayLike, unit: Any, /, **kw: Any
) -> DistanceModulus:
    """Construct a distance.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> cxd.DistanceModulus.from_(1, "mag")
    DistanceModulus(1, 'mag')

    """
    return cls(jnp.asarray(value, **kw), unit)


@DistanceModulus.from_.dispatch
def from_(
    cls: type[DistanceModulus], dm: DistanceModulus, /, **kw: Any
) -> DistanceModulus:
    """Compute distance modulus from distance modulus.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> dm = cxd.DistanceModulus(1, "mag")
    >>> cxd.DistanceModulus.from_(dm) is dm
    True

    >>> cxd.DistanceModulus.from_(dm, dtype=float)
    DistanceModulus(1., 'mag')

    """
    if len(kw) == 0:
        return dm
    return jnp.asarray(dm, **kw)


@DistanceModulus.from_.dispatch
def from_(cls: type[DistanceModulus], dm: u.Q["mag"], /, **kw: Any) -> DistanceModulus:  # type: ignore[type-arg]
    """Compute parallax from parallax.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> q = u.Q(1, "mag")
    >>> cxd.DistanceModulus.from_(q)
    DistanceModulus(1, 'mag')

    """
    unit = u.unit_of(dm)
    return cls(jnp.asarray(u.ustrip(unit, dm), **kw), unit)


@DistanceModulus.from_.dispatch
def from_(
    cls: type[DistanceModulus],
    d: Distance,
    /,
    **kw: Any,
) -> DistanceModulus:
    """Compute distance modulus from distance.

    Examples
    --------
    >>> import coordinax.distances as cxd

    >>> d = cxd.Distance(1, "pc")
    >>> cxd.DistanceModulus.from_(d)
    DistanceModulus(-5., 'mag')

    """
    dm = 5 * jnp.log10(d.ustrip("pc")) - 5
    return cls(jnp.asarray(dm, **kw), "mag")


@DistanceModulus.from_.dispatch
def from_(
    cls: type[DistanceModulus],
    d: u.Q["length"],  # type: ignore[type-arg]
    /,
    **kw: Any,
) -> DistanceModulus:
    """Compute distance modulus from distance.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> q = u.Q(1, "pc")
    >>> cxd.DistanceModulus.from_(q)
    DistanceModulus(-5., 'mag')

    """
    dm = 5 * jnp.log10(d.ustrip("pc")) - 5
    return cls(jnp.asarray(dm, **kw), "mag")


@DistanceModulus.from_.dispatch
def from_(
    cls: type[DistanceModulus],
    p: "Parallax",  # type: ignore[type-arg]
    /,
    **kw: Any,
) -> DistanceModulus:
    """Compute distance modulus from parallax.

    Examples
    --------
    >>> import coordinax.distances as cxd

    >>> p = cxd.Parallax(1, "mas")
    >>> cxd.DistanceModulus.from_(p)
    DistanceModulus(10., 'mag')

    """
    d = parallax_base_length / jnp.tan(p)  # [AU]
    dm = 5 * jnp.log10(d.ustrip("pc")) - 5
    return cls(jnp.asarray(dm, **kw), "mag")


@DistanceModulus.from_.dispatch
def from_(
    cls: type[DistanceModulus],
    p: u.Q["angle"],  # type: ignore[type-arg]
    /,
    **kw: Any,
) -> DistanceModulus:
    """Compute distance modulus from parallax.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> q = u.Q(1, "mas")
    >>> cxd.DistanceModulus.from_(q)
    DistanceModulus(10., 'mag')

    """
    d = parallax_base_length / jnp.tan(p)  # [AU]
    dm = 5 * jnp.log10(d.ustrip("pc")) - 5
    return cls(jnp.asarray(dm, **kw), "mag")


##############################################################################


@final
class Parallax(AbstractDistance):
    """Parallax distance quantity.

    Examples
    --------
    >>> from coordinax.distances import Parallax
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

    unit: u.AbstractUnit = eqx.field(static=True, converter=u.unit)
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


@Parallax.from_.dispatch
def from_(cls: type[Parallax], value: ArrayLike, unit: Any, /, **kw: Any) -> Parallax:
    """Construct a distance.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> cxd.Parallax.from_(1, "mas")
    Parallax(1, 'mas')

    """
    return cls(jnp.asarray(value, **kw), unit)


@Parallax.from_.dispatch
def from_(cls: type[Parallax], p: Parallax, /, **kw: Any) -> Parallax:
    """Compute parallax from parallax.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> p = cxd.Parallax(1, "mas")
    >>> cxd.Parallax.from_(p) is p
    True

    >>> cxd.Parallax.from_(p, dtype=float)
    Parallax(1., 'mas')

    """
    if len(kw) == 0:
        return p
    return jnp.asarray(p, **kw)


@Parallax.from_.dispatch
def from_(cls: type[Parallax], p: u.Q["angle"], /, **kw: Any) -> Parallax:  # type: ignore[type-arg]
    """Compute parallax from parallax.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> q = u.Q(1, "mas")
    >>> cxd.Parallax.from_(q, dtype=float)
    Parallax(1., 'mas')

    """
    unit = u.unit_of(p)
    return cls(jnp.asarray(p.ustrip(unit), **kw), unit)


@Parallax.from_.dispatch
def from_(cls: type[Parallax], d: Distance | u.Q["length"], /, **kw: Any) -> Parallax:  # type: ignore[type-arg]
    """Compute parallax from distance.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> d = cxd.Distance(10, "pc")
    >>> cxd.Parallax.from_(d).uconvert("mas").round(2)
    Parallax(100., 'mas')

    >>> q = u.Q(10, "pc")
    >>> cxd.Parallax.from_(q).uconvert("mas").round(2)
    Parallax(100., 'mas')

    """
    p = jnp.atan2(parallax_base_length, d)
    return cls(jnp.asarray(p.value, **kw), p.unit)


@Parallax.from_.dispatch
def from_(
    cls: type[Parallax],
    dm: DistanceModulus | u.Q["mag"],  # type: ignore[type-arg]
    /,
    **kw: Any,
) -> Parallax:
    """Convert distance modulus to parallax.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distances as cxd

    >>> dm = cxd.DistanceModulus(10, "mag")
    >>> cxd.Parallax.from_(dm).uconvert("mas").round(2)
    Parallax(1., 'mas')

    """
    d = BareQuantity(10 ** (1 + dm.ustrip("mag") / 5), "pc")
    p = jnp.atan2(parallax_base_length, d)
    unit = u.unit_of(p)
    return cls(jnp.asarray(p.ustrip(unit), **kw), unit)
