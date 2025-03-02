"""distance functions."""

__all__ = [
    "parallax",
    "distance_modulus",
]


from typing import Any

from plum import dispatch

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import BareQuantity

from .measures import Distance, DistanceModulus, Parallax

parallax_base_length = u.Quantity(1, "AU")


#####################################################################
# Parallax


@dispatch
def parallax(p: Parallax, /, **kwargs: Any) -> Parallax:
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
    if len(kwargs) == 0:
        return p
    return jnp.asarray(p, **kwargs)


@dispatch
def parallax(p: u.Quantity["angle"], /) -> Parallax:
    """Compute parallax from parallax.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distance as cxd

    >>> q = u.Quantity(1, "mas")
    >>> cxd.parallax(q)
    Parallax(Array(1, dtype=int32, weak_type=True), unit='mas')

    """
    unit = u.unit_of(p)
    return Parallax(p.ustrip(unit), unit)


@dispatch
def parallax(d: Distance | u.Quantity["length"], /, **kwargs: Any) -> Parallax:
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
    return Parallax(jnp.asarray(p.value, **kwargs), p.unit)


@dispatch
def parallax(dm: DistanceModulus | u.Quantity["mag"], /, **kwargs: Any) -> Parallax:
    """Convert distance modulus to parallax.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distance as cxd

    >>> dm = cxd.DistanceModulus(10, "mag")
    >>> cxd.parallax(dm).uconvert("mas").round(2)
    Parallax(Array(1., dtype=float32, weak_type=True), unit='mas')

    """
    d = BareQuantity(10 ** (1 + dm.ustrip("mag") / 5), "pc")
    p = jnp.atan2(parallax_base_length, d)
    unit = u.unit_of(p)
    return Parallax(jnp.asarray(p.ustrip(unit), **kwargs), unit)


#####################################################################
# Distance Modulus


@dispatch
def distance_modulus(dm: DistanceModulus, /, **kwargs: Any) -> DistanceModulus:
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
    if len(kwargs) == 0:
        return dm
    return jnp.asarray(dm, **kwargs)


@dispatch
def distance_modulus(dm: u.Quantity["mag"], /) -> DistanceModulus:
    """Compute parallax from parallax.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.distance as cxd

    >>> q = u.Quantity(1, "mag")
    >>> cxd.distance_modulus(q)
    DistanceModulus(Array(1, dtype=int32, weak_type=True), unit='mag')

    """
    unit = u.unit_of(dm)
    return DistanceModulus(u.ustrip(unit, dm), unit)


@dispatch
def distance_modulus(
    d: Distance | u.Quantity["length"], /, **kwargs: Any
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
    return DistanceModulus(jnp.asarray(dm, **kwargs), "mag")


@dispatch
def distance_modulus(
    p: Parallax | u.Quantity["angle"], /, **kwargs: Any
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
    return DistanceModulus(jnp.asarray(dm, **kwargs), "mag")
