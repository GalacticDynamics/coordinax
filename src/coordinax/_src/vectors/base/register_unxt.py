"""Register Support with Unxt."""

__all__ = ("ToUnitsOptions",)


from enum import Enum

from collections.abc import Mapping
from typing import Any, Literal

from astropy.units import PhysicalType as Dimension
from plum import dispatch

import unxt as u
from dataclassish import field_items, replace

from .flags import AttrFilter
from .vector import AbstractVector
from coordinax._src.custom_types import Unit


class ToUnitsOptions(Enum):
    """Options for the units argument of `AbstractVector.uconvert`."""

    consistent = "consistent"
    """Convert to consistent units."""


@dispatch
def uconvert(usys: u.AbstractUnitSystem, vector: AbstractVector, /) -> AbstractVector:
    """Convert the vector to the given units.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> usys = u.unitsystem("m", "s", "kg", "rad")

    >>> vec = cx.CartesianPos3D.from_([1, 2, 3], "km")
    >>> print(u.uconvert(usys, vec))
    <CartesianPos3D: (x, y, z) [m]
        [1000. 2000. 3000.]>

    """
    usys = u.unitsystem(usys)
    return replace(
        vector,
        **{
            k: u.uconvert(usys[u.dimension_of(v)], v)
            for k, v in field_items(AttrFilter, vector)
        },
    )


@dispatch
def uconvert(
    units: Mapping[Dimension, Unit | str], vector: AbstractVector, /
) -> AbstractVector:
    """Convert the vector to the given units.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    We can convert a vector to the given units:

    >>> cart = cx.vecs.CartesianPos2D(x=u.Q(1, "m"), y=u.Q(2, "km"))
    >>> cart.uconvert({u.dimension("length"): "km"})
    CartesianPos2D(x=Q(0.001, 'km'), y=Q(2, 'km'))

    This also works for vectors with different units:

    >>> sph = cx.SphericalPos(r=u.Q(1, "m"), theta=u.Q(45, "deg"),
    ...                       phi=u.Q(3, "rad"))
    >>> sph.uconvert({u.dimension("length"): "km", u.dimension("angle"): "deg"})
    SphericalPos(
      r=Distance(0.001, 'km'), theta=Angle(45, 'deg'), phi=Angle(171.88734, 'deg')
    )

    """
    # Ensure `units_` is PT -> Unit
    units_ = {u.dimension(k): v for k, v in units.items()}
    # Convert to the given units
    return replace(
        vector,
        **{
            k: u.uconvert(units_[u.dimension_of(v)], v)
            for k, v in field_items(AttrFilter, vector)
        },
    )


@dispatch
def uconvert(units: Mapping[str, Any], vector: AbstractVector, /) -> AbstractVector:
    """Convert the vector to the given units.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    We can convert a vector to the given units:

    >>> cart = cx.vecs.CartesianPos2D(x=u.Q(1, "m"), y=u.Q(2, "km"))
    >>> cart.uconvert({"x": "km", "y": "m"})
    CartesianPos2D(x=Q(0.001, 'km'), y=Q(2000., 'm'))

    This also works for converting just some of the components:

    >>> cart.uconvert({"x": "km"})
    CartesianPos2D(x=Q(0.001, 'km'), y=Q(2, 'km'))

    This also works for vectors with different units:

    >>> sph = cx.SphericalPos(r=u.Q(1, "m"), theta=u.Q(45, "deg"),
    ...                       phi=u.Q(3, "rad"))
    >>> sph.uconvert({"r": "km", "theta": "rad"})
    SphericalPos( r=Distance(0.001, 'km'), theta=Angle(0.7853982, 'rad'),
                  phi=Angle(3., 'rad') )

    """
    return replace(
        vector,
        **{
            k: u.uconvert(units.get(k, u.unit_of(v)), v)
            for k, v in field_items(AttrFilter, vector)
        },
    )


@dispatch
def uconvert(
    flag: Literal[ToUnitsOptions.consistent], vector: AbstractVector, /
) -> AbstractVector:
    """Convert the vector to a self-consistent set of units.

    Parameters
    ----------
    flag
        The vector is converted to consistent units by looking for the first
        quantity with each physical type and converting all components to
        the units of that quantity.
    vector
        The vector to convert.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    We can convert a vector to the given units:

    >>> cart = cx.vecs.CartesianPos2D.from_([1, 2], "m")

    If all you want is to convert to consistent units, you can use
    ``"consistent"``:

    >>> cart.uconvert(cx.vecs.ToUnitsOptions.consistent)
    CartesianPos2D(x=Q(1, 'm'), y=Q(2, 'm'))

    >>> sph = cart.vconvert(cx.SphericalPos)
    >>> sph.uconvert(cx.vecs.ToUnitsOptions.consistent)
    SphericalPos( r=Distance(2.236068, 'm'), theta=Angle(1.5707964, 'rad'),
                  phi=Angle(1.1071488, 'rad') )

    """
    dim2unit = {}
    units_ = {}
    for k, v in field_items(AttrFilter, vector):
        pt = u.dimension_of(v)
        if pt not in dim2unit:  # pylint: disable=unreachable
            dim2unit[pt] = u.unit_of(v)
        units_[k] = dim2unit[pt]

    return replace(
        vector,
        **{k: u.uconvert(units_[k], v) for k, v in field_items(AttrFilter, vector)},
    )


@dispatch
def uconvert(usys: str, vector: AbstractVector, /) -> AbstractVector:
    """Convert the vector to the given units system.

    Parameters
    ----------
    usys
        The units system to convert to, as a string.
    vector
        The vector to convert.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> usys = "galactic"
    >>> vector = cx.CartesianPos3D.from_([1, 2, 3], "m")
    >>> u.uconvert(usys, vector)
    CartesianPos3D( x=Q(3.2407793e-20, 'kpc'), y=Q(6.4815585e-20, 'kpc'),
                    z=Q(9.722338e-20, 'kpc') )

    """
    usys = u.unitsystem(usys)
    return uconvert(usys, vector)
