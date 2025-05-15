"""distance functions."""

__all__ = [
    "distance",
    "parallax",
    "distance_modulus",
]


from typing import Any

from jaxtyping import ArrayLike
from plum import dispatch

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import BareQuantity

from .measures import Distance, DistanceModulus, Parallax

parallax_base_length = u.Quantity(1, "AU")

#####################################################################
# Distance constructor


@dispatch
def distance(value: ArrayLike, unit: Any, /, **kw: Any) -> Distance:
    """Construct a distance.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distance as cxd

    >>> cxd.distance(1, "kpc")
    Distance(Array(1, dtype=int32, ...), unit='kpc')

    """
    return Distance(jnp.asarray(value, **kw), unit)


@dispatch
def distance(d: Distance, /, **kw: Any) -> Distance:
    """Compute distance from distance.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distance as cxd

    >>> d = cxd.Distance(1, "kpc")
    >>> cxd.distance(d) is d
    True

    >>> cxd.distance(d, dtype=float)
    Distance(Array(1., dtype=float32), unit='kpc')

    """
    if len(kw) == 0:
        return d
    return jnp.asarray(d, **kw)


@dispatch
def distance(d: u.Quantity["length"], /, **kw: Any) -> Distance:
    """Compute distance from distance.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distance as cxd

    >>> q = u.Quantity(1, "kpc")
    >>> cxd.distance(q, dtype=float)
    Distance(Array(1., dtype=float32), unit='kpc')

    """
    unit = u.unit_of(d)
    return Distance(jnp.asarray(d.ustrip(unit), **kw), unit)


@dispatch
def distance(p: Parallax | u.Quantity["angle"], /, **kw: Any) -> Distance:
    """Compute distance from parallax.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distance as cxd

    >>> p = cxd.Parallax(1, "mas")
    >>> cxd.distance(p).uconvert("pc").round(2)
    Distance(Array(1000., dtype=float32, ...), unit='pc')

    >>> q = u.Quantity(1, "mas")
    >>> cxd.distance(q).uconvert("pc").round(2)
    Distance(Array(1000., dtype=float32, ...), unit='pc')

    """
    d = parallax_base_length / jnp.tan(p)  # [AU]
    unit = u.unit_of(d)
    return Distance(jnp.asarray(d.ustrip(unit), **kw), unit)


@dispatch
def distance(dm: DistanceModulus | u.Quantity["mag"], /, **kw: Any) -> Distance:
    """Compute distance from distance modulus.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distance as cxd

    >>> dm = cxd.DistanceModulus(10, "mag")
    >>> cxd.distance(dm).uconvert("pc").round(2)
    Distance(Array(1000., dtype=float32, ...), unit='pc')

    >>> q = u.Quantity(10, "mag")
    >>> cxd.distance(q).uconvert("pc").round(2)
    Distance(Array(1000., dtype=float32, ...), unit='pc')

    """
    d = 10 ** (1 + dm.ustrip("mag") / 5)
    return Distance(jnp.asarray(d, **kw), "pc")


#####################################################################
# Parallax constructor


@dispatch
def parallax(value: ArrayLike, unit: Any, /, **kw: Any) -> Parallax:
    """Construct a distance.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distance as cxd

    >>> cxd.parallax(1, "mas")
    Parallax(Array(1, dtype=int32, ...), unit='mas')

    """
    return Parallax(jnp.asarray(value, **kw), unit)


@dispatch
def parallax(p: Parallax, /, **kw: Any) -> Parallax:
    """Compute parallax from parallax.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distance as cxd

    >>> p = cxd.Parallax(1, "mas")
    >>> cxd.parallax(p) is p
    True

    >>> cxd.parallax(p, dtype=float)
    Parallax(Array(1., dtype=float32), unit='mas')

    """
    if len(kw) == 0:
        return p
    return jnp.asarray(p, **kw)


@dispatch
def parallax(p: u.Quantity["angle"], /, **kw: Any) -> Parallax:
    """Compute parallax from parallax.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distance as cxd

    >>> q = u.Quantity(1, "mas")
    >>> cxd.parallax(q, dtype=float)
    Parallax(Array(1., dtype=float32), unit='mas')

    """
    unit = u.unit_of(p)
    return Parallax(jnp.asarray(p.ustrip(unit), **kw), unit)


@dispatch
def parallax(d: Distance | u.Quantity["length"], /, **kw: Any) -> Parallax:
    """Compute parallax from distance.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distance as cxd

    >>> d = cxd.Distance(10, "pc")
    >>> cxd.parallax(d).uconvert("mas").round(2)
    Parallax(Array(100., dtype=float32, ...), unit='mas')

    >>> q = u.Quantity(10, "pc")
    >>> cxd.parallax(q).uconvert("mas").round(2)
    Parallax(Array(100., dtype=float32, ...), unit='mas')

    """
    p = jnp.atan2(parallax_base_length, d)
    return Parallax(jnp.asarray(p.value, **kw), p.unit)


@dispatch
def parallax(dm: DistanceModulus | u.Quantity["mag"], /, **kw: Any) -> Parallax:
    """Convert distance modulus to parallax.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distance as cxd

    >>> dm = cxd.DistanceModulus(10, "mag")
    >>> cxd.parallax(dm).uconvert("mas").round(2)
    Parallax(Array(1., dtype=float32, ...), unit='mas')

    """
    d = BareQuantity(10 ** (1 + dm.ustrip("mag") / 5), "pc")
    p = jnp.atan2(parallax_base_length, d)
    unit = u.unit_of(p)
    return Parallax(jnp.asarray(p.ustrip(unit), **kw), unit)


#####################################################################
# Distance Modulus constructor


@dispatch
def distance_modulus(value: ArrayLike, unit: Any, /, **kw: Any) -> DistanceModulus:
    """Construct a distance.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distance as cxd

    >>> cxd.distance_modulus(1, "mag")
    DistanceModulus(Array(1, dtype=int32, ...), unit='mag')

    """
    return DistanceModulus(jnp.asarray(value, **kw), unit)


@dispatch
def distance_modulus(dm: DistanceModulus, /, **kw: Any) -> DistanceModulus:
    """Compute distance modulus from distance modulus.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distance as cxd

    >>> dm = cxd.DistanceModulus(1, "mag")
    >>> cxd.distance_modulus(dm) is dm
    True

    >>> cxd.distance_modulus(dm, dtype=float)
    DistanceModulus(Array(1., dtype=float32), unit='mag')

    """
    if len(kw) == 0:
        return dm
    return jnp.asarray(dm, **kw)


@dispatch
def distance_modulus(dm: u.Quantity["mag"], /, **kw: Any) -> DistanceModulus:
    """Compute parallax from parallax.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distance as cxd

    >>> q = u.Quantity(1, "mag")
    >>> cxd.distance_modulus(q)
    DistanceModulus(Array(1, dtype=int32, ...), unit='mag')

    """
    unit = u.unit_of(dm)
    return DistanceModulus(jnp.asarray(u.ustrip(unit, dm), **kw), unit)


@dispatch
def distance_modulus(
    d: Distance | u.Quantity["length"], /, **kw: Any
) -> DistanceModulus:
    """Compute distance modulus from distance.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distance as cxd

    >>> d = cxd.Distance(1, "pc")
    >>> cxd.distance_modulus(d)
    DistanceModulus(Array(-5., dtype=float32), unit='mag')

    >>> q = u.Quantity(1, "pc")
    >>> cxd.distance_modulus(q)
    DistanceModulus(Array(-5., dtype=float32), unit='mag')

    """
    dm = 5 * jnp.log10(d.ustrip("pc")) - 5
    return DistanceModulus(jnp.asarray(dm, **kw), "mag")


@dispatch
def distance_modulus(
    p: Parallax | u.Quantity["angle"], /, **kw: Any
) -> DistanceModulus:
    """Compute distance modulus from parallax.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distance as cxd

    >>> p = cxd.Parallax(1, "mas")
    >>> cxd.distance_modulus(p)
    DistanceModulus(Array(10., dtype=float32), unit='mag')

    >>> q = u.Quantity(1, "mas")
    >>> cxd.distance_modulus(q)
    DistanceModulus(Array(10., dtype=float32), unit='mag')

    """
    d = parallax_base_length / jnp.tan(p)  # [AU]
    dm = 5 * jnp.log10(d.ustrip("pc")) - 5
    return DistanceModulus(jnp.asarray(dm, **kw), "mag")
