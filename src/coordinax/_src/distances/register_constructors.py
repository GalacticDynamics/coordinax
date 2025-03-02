"""Distance constructors."""

__all__: list[str] = []

from typing import Any

import jax.numpy as jnp

import quaxed.numpy as jnp
import unxt as u

from .base import AbstractDistance
from .funcs import parallax
from .measures import Distance, DistanceModulus, Parallax

parallax_base_length = u.Quantity(1, "AU")
distance_modulus_base_distance = u.Quantity(10, "pc")

# -------------------------------------------------------------------
# To Distance


@u.AbstractQuantity.from_.dispatch
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


@u.AbstractQuantity.from_.dispatch
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


@u.AbstractQuantity.from_.dispatch  # type: ignore[no-redef]
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


@u.AbstractQuantity.from_.dispatch
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


@u.AbstractQuantity.from_.dispatch
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


@u.AbstractQuantity.from_.dispatch  # type: ignore[no-redef]
def from_(
    cls: type[DistanceModulus], p: Parallax | u.Quantity["angle"], /, **kwargs: Any
) -> DistanceModulus:
    """Construct a `DistanceModulus` from a parallax.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distance as cxd

    >>> p = cxd.Parallax(1, "mas")
    >>> cxd.DistanceModulus.from_(p)
    DistanceModulus(Array(10., dtype=float32), unit='mag')

    >>> q = u.Quantity(1, "mas")
    >>> DistanceModulus.from_(q)
    DistanceModulus(Array(10., dtype=float32), unit='mag')

    """
    d = parallax_base_length / jnp.tan(p)  # [AU]
    dm = 5 * jnp.log10(d.ustrip("pc")) - 5
    return cls(jnp.asarray(dm, **kwargs), "mag")


# -------------------------------------------------------------------
# To Parallax


@u.AbstractQuantity.from_.dispatch
def from_(
    cls: type[Parallax],
    obj: AbstractDistance
    | u.Quantity["angle"]
    | u.Quantity["length"]
    | u.Quantity["mag"],
    /,
    **kwargs: Any,
) -> Parallax:
    """Construct a `Parallax` the input."""
    return parallax(obj, **kwargs)
