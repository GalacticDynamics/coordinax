"""distance functions."""

__all__ = ["parallax"]


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
