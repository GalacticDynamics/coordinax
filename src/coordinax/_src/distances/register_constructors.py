"""Distance constructors."""

__all__: list[str] = []

from typing import Any

import jax.numpy as jnp

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AbstractQuantity, UncheckedQuantity

from .measures import Distance, DistanceModulus, Parallax

parallax_base_length = u.Quantity(1, "AU")
distance_modulus_base_distance = u.Quantity(10, "pc")

# -------------------------------------------------------------------
# To Distance


@AbstractQuantity.from_.dispatch
def from_(cls: type[Distance], d: Distance) -> Distance:
    """Construct a `Distance` from a `Distance`.

    Examples
    --------
    >>> from coordinax.distance import Distance

    >>> d = Distance(1, "pc")
    >>> Distance.from_(d) is d
    True

    """
    return d


@AbstractQuantity.from_.dispatch
def from_(
    cls: type[Distance], p: Parallax | u.Quantity["angle"], /, **kwargs: Any
) -> Distance:
    """Construct a `coordinax.Distance` from an angle through the parallax.

    Examples
    --------
    >>> import unxt as u
    >>> from coordinax.distance import Distance, Parallax

    >>> Distance.from_(Parallax(1, "mas")).uconvert("kpc")
    Distance(Array(1., dtype=float32, ...), unit='kpc')

    >>> Distance.from_(u.Quantity(1, "mas")).uconvert("kpc")
    Distance(Array(1., dtype=float32, ...), unit='kpc')

    """
    d = parallax_base_length / jnp.tan(p)  # [AU]
    unit = u.unit_of(d)
    return cls(jnp.asarray(d.ustrip(unit), **kwargs), unit)


@AbstractQuantity.from_.dispatch  # type: ignore[no-redef]
def from_(
    cls: type[Distance], dm: DistanceModulus | u.Quantity["mag"], /, **kwargs: Any
) -> Distance:
    """Construct a `Distance` from a mag through the dist mod.

    Examples
    --------
    >>> import unxt as u
    >>> from coordinax.distance import Distance, DistanceModulus

    >>> Distance.from_(DistanceModulus(10, "mag")).uconvert("pc")
    Distance(Array(1000., dtype=float32, ...), unit='pc')

    >>> Distance.from_(u.Quantity(10, "mag")).uconvert("pc")
    Distance(Array(1000., dtype=float32, ...), unit='pc')

    """
    d = 10 ** (1 + dm.ustrip("mag") / 5)
    return cls(jnp.asarray(d, **kwargs), "pc")


# -------------------------------------------------------------------
# To DistanceModulus


@AbstractQuantity.from_.dispatch
def from_(cls: type[DistanceModulus], dm: DistanceModulus) -> DistanceModulus:
    """Construct a `DistanceModulus` from a `DistanceModulus`.

    Examples
    --------
    >>> from coordinax.distance import DistanceModulus

    >>> dm = DistanceModulus(1, "mag")
    >>> DistanceModulus.from_(dm) is dm
    True

    """
    return dm


@AbstractQuantity.from_.dispatch
def from_(
    cls: type[DistanceModulus], d: Distance | u.Quantity["length"], /, **kwargs: Any
) -> DistanceModulus:
    """Construct a `DistanceModulus` from a distance.

    Examples
    --------
    >>> import unxt as u
    >>> from coordinax.distance import Distance, DistanceModulus

    >>> DistanceModulus.from_(Distance(1, "pc"))
    DistanceModulus(Array(-5., dtype=float32), unit='mag')

    >>> DistanceModulus.from_(u.Quantity(1, "pc"))
    DistanceModulus(Array(-5., dtype=float32), unit='mag')

    """
    dm = 5 * jnp.log10(d.ustrip("pc")) - 5
    return cls(jnp.asarray(dm, **kwargs), "mag")


@AbstractQuantity.from_.dispatch  # type: ignore[no-redef]
def from_(
    cls: type[DistanceModulus], p: Parallax | u.Quantity["angle"], /, **kwargs: Any
) -> DistanceModulus:
    """Construct a `DistanceModulus` from a parallax.

    Examples
    --------
    >>> import unxt as u
    >>> from coordinax.distance import DistanceModulus, Parallax

    >>> DistanceModulus.from_(Parallax(1, "mas"))
    DistanceModulus(Array(10., dtype=float32), unit='mag')

    >>> DistanceModulus.from_(u.Quantity(1, "mas"))
    DistanceModulus(Array(10., dtype=float32), unit='mag')

    """
    d = parallax_base_length / jnp.tan(p)  # [AU]
    dm = 5 * jnp.log10(d.ustrip("pc")) - 5
    return cls(jnp.asarray(dm, **kwargs), "mag")


# -------------------------------------------------------------------
# To Parallax


@AbstractQuantity.from_.dispatch
def from_(cls: type[Parallax], p: Parallax) -> Parallax:
    """Construct a `Parallax` from a `Parallax`.

    Examples
    --------
    >>> from coordinax.distance import Parallax

    >>> p = Parallax(1, "mas")
    >>> Parallax.from_(p) is p
    True

    """
    return p


@AbstractQuantity.from_.dispatch  # type: ignore[no-redef]
def from_(
    cls: type[Parallax], d: Distance | u.Quantity["length"], /, **kwargs: Any
) -> Parallax:
    """Construct a `Parallax` from a distance.

    Examples
    --------
    >>> import unxt as u
    >>> from coordinax.distance import Parallax, Distance

    >>> Parallax.from_(Distance(10, "pc")).uconvert("mas").round(2)
    Parallax(Array(100., dtype=float32, ...), unit='mas')

    >>> Parallax.from_(u.Quantity(10, "pc")).uconvert("mas").round(2)
    Parallax(Array(100., dtype=float32, ...), unit='mas')

    """
    p = jnp.atan2(parallax_base_length, d)
    return cls(jnp.asarray(p.value, **kwargs), p.unit)


@AbstractQuantity.from_.dispatch  # type: ignore[no-redef]
def from_(
    cls: type[Parallax], dm: DistanceModulus | u.Quantity["mag"], /, **kwargs: Any
) -> Parallax:
    """Construct a `Parallax` from a distance.

    Examples
    --------
    >>> import unxt as u
    >>> from coordinax.distance import Parallax, DistanceModulus

    >>> Parallax.from_(DistanceModulus(23, "mag")).uconvert("mas").round(2)
    Parallax(Array(0., dtype=float32, ...), unit='mas')

    >>> Parallax.from_(u.Quantity(23, "mag")).uconvert("mas").round(2)
    Parallax(Array(0., dtype=float32, ...), unit='mas')

    """
    d = UncheckedQuantity(10 ** (1 + dm.ustrip("mag") / 5), "pc")
    p = jnp.atan2(parallax_base_length, d)
    unit = u.unit_of(p)
    return cls(jnp.asarray(p.ustrip(unit), **kwargs), unit)
