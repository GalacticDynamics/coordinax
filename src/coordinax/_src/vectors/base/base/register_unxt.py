"""Register Support with Unxt."""

__all__: list[str] = []


from collections.abc import Mapping
from typing import Any, Literal

from astropy.units import PhysicalType as Dimension
from plum import dispatch

import unxt as u
from dataclassish import field_items, replace

from .flags import AttrFilter
from .utils import ToUnitsOptions
from .vector import AbstractVector
from coordinax._src.typing import Unit


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
    <CartesianPos3D (x[m], y[m], z[m])
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

    >>> cart = cx.vecs.CartesianPos2D(x=u.Quantity(1, "m"), y=u.Quantity(2, "km"))
    >>> cart.uconvert({u.dimension("length"): "km"})
    CartesianPos2D(
        x=Quantity[...](value=...f32[], unit=Unit("km")),
        y=Quantity[...](value=...i32[], unit=Unit("km"))
    )

    This also works for vectors with different units:

    >>> sph = cx.SphericalPos(r=u.Quantity(1, "m"), theta=u.Quantity(45, "deg"),
    ...                       phi=u.Quantity(3, "rad"))
    >>> sph.uconvert({u.dimension("length"): "km", u.dimension("angle"): "deg"})
    SphericalPos(
      r=Distance(value=...f32[], unit=Unit("km")),
      theta=Angle(value=...i32[], unit=Unit("deg")),
      phi=Angle(value=...f32[], unit=Unit("deg")) )

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

    >>> cart = cx.vecs.CartesianPos2D(x=u.Quantity(1, "m"), y=u.Quantity(2, "km"))
    >>> cart.uconvert({"x": "km", "y": "m"})
    CartesianPos2D(
        x=Quantity[...](value=...f32[], unit=Unit("km")),
        y=Quantity[...](value=...f32[], unit=Unit("m"))
    )

    This also works for converting just some of the components:

    >>> cart.uconvert({"x": "km"})
    CartesianPos2D(
        x=Quantity[...](value=...f32[], unit=Unit("km")),
        y=Quantity[...](value=...i32[], unit=Unit("km"))
    )

    This also works for vectors with different units:

    >>> sph = cx.SphericalPos(r=u.Quantity(1, "m"), theta=u.Quantity(45, "deg"),
    ...                       phi=u.Quantity(3, "rad"))
    >>> sph.uconvert({"r": "km", "theta": "rad"})
    SphericalPos(
        r=Distance(value=...f32[], unit=Unit("km")),
        theta=Angle(value=...f32[], unit=Unit("rad")),
        phi=Angle(value=...f32[], unit=Unit("rad")) )

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
) -> "AbstractVector":
    """Convert the vector to a self-consistent set of units.

    Parameters
    ----------
    flag : Literal[ToUnitsOptions.consistent]
        The vector is converted to consistent units by looking for the first
        quantity with each physical type and converting all components to
        the units of that quantity.
    vector : AbstractVector
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
    CartesianPos2D(
        x=Quantity[...](value=i32[], unit=Unit("m")),
        y=Quantity[...](value=i32[], unit=Unit("m"))
    )

    >>> sph = cart.vconvert(cx.SphericalPos)
    >>> sph.uconvert(cx.vecs.ToUnitsOptions.consistent)
    SphericalPos(
        r=Distance(value=f32[], unit=Unit("m")),
        theta=Angle(value=f32[], unit=Unit("rad")),
        phi=Angle(value=f32[], unit=Unit("rad"))
    )

    """
    dim2unit = {}
    units_ = {}
    for k, v in field_items(AttrFilter, vector):
        pt = u.dimension_of(v)
        if pt not in dim2unit:
            dim2unit[pt] = u.unit_of(v)
        units_[k] = dim2unit[pt]

    return replace(
        vector,
        **{k: u.uconvert(units_[k], v) for k, v in field_items(AttrFilter, vector)},
    )
