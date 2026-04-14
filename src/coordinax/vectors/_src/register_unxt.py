"""Point."""

__all__ = ("ToUnitsOptions",)

from dataclasses import replace
from enum import Enum

from collections.abc import Mapping
from typing import Any, Literal, cast, final

import plum

import unxt as u

from .point import Point
from coordinax.internal import QuantityMatrix, pack_nonuniform_unit


@final
class ToUnitsOptions(Enum):
    """Options for the units argument of `uconvert`.

    This enum provides named conversion behaviors that are accepted by
    `Point.uconvert` and `Coordinate.uconvert`.

    Examples
    --------
    `Point.uconvert` with `consistent`:

    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> vec = cx.Point.from_({"x": u.Q(1, "m"), "y": u.Q(2, "km")}, cx.cart2d)
    >>> print(vec.uconvert(cx.ToUnitsOptions.consistent))
    <Point: chart=Cart2D (x, y) [m]
        [1.e+00 2.e+03]>

    """

    consistent = "consistent"
    """Convert to consistent units."""


@plum.dispatch
def uconvert(usys: u.AbstractUnitSystem, vec: Point, /) -> Point:
    """Convert the point to the given units.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> usys = u.unitsystem("m", "s", "kg", "rad")

    >>> vec = cx.Point.from_([1, 2, 3], "km")
    >>> print(u.uconvert(usys, vec))
    <Point: chart=Cart3D (x, y, z) [m]
        [1000. 2000. 3000.]>

    """
    data = {k: u.uconvert(usys[u.dimension_of(v)], v) for k, v in vec.data.items()}
    return replace(vec, data=data)


@plum.dispatch
def uconvert(
    units: Mapping[u.AbstractDimension, u.AbstractUnit | str], vec: Point, /
) -> Point:
    """Convert the point to the given units.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.main as cx

    We can convert a point to the given units:

    >>> cart = cx.Point.from_({"x": u.Q(1, "m"), "y": u.Q(2, "km")}, cx.cart2d)
    >>> print(cart.uconvert({u.dimension("length"): "km"}))
    <Point: chart=Cart2D (x, y) [km]
        [1.e-03 2.e+00]>

    This also works for points with different units:

    >>> sph = cx.Point.from_(
    ...     {"r": u.Q(1, "m"), "theta": u.Q(45, "deg"), "phi": u.Q(3, "rad")},
    ...     cx.sph3d)
    >>> print(sph.uconvert({u.dimension("length"): "km", u.dimension("angle"): "deg"}))
    <Point: chart=Spherical3D (r[km], theta[deg], phi[deg])
        [1.000e-03 4.500e+01 1.719e+02]>

    """
    # # Ensure `units_` is PT -> Unit
    units_ = {u.dimension(k): v for k, v in units.items()}
    data = {k: u.uconvert(units_[u.dimension_of(v)], v) for k, v in vec.data.items()}
    return replace(vec, data=data)


@plum.dispatch
def uconvert(units: Mapping[str, Any], vec: Point, /) -> Point:
    """Convert the point to the given units.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.main as cx

    We can convert a point to the given units:

    >>> cart = cx.Point.from_({"x": u.Q(1, "m"), "y": u.Q(2, "km")}, cx.cart2d)
    >>> print(cart.uconvert({"x": "km", "y": "m"}))
    <Point: chart=Cart2D (x[km], y[m])
        [1.e-03 2.e+03]>

    This also works for converting just some of the components:

    >>> print(cart.uconvert({"x": "km"}))
    <Point: chart=Cart2D (x, y) [km]
        [1.e-03 2.e+00]>

    This also works for points with different units:

    >>> sph = cx.Point.from_({"r": u.Q(1, "m"), "theta": u.Q(45, "deg"),
    ...                       "phi": u.Q(3, "rad")}, cx.sph3d)
    >>> print(sph.uconvert({"r": "km", "theta": "rad"}))
    <Point: chart=Spherical3D (r[km], theta[rad], phi[rad])
        [1.000e-03 7.854e-01 3.000e+00]>

    """
    data = {  # (component: unit)
        k: u.uconvert(units.get(k, u.unit_of(v)), v)  # default to original unit
        for k, v in vec.data.items()
    }
    return replace(vec, data=data)


@plum.dispatch
def uconvert(flag: Literal[ToUnitsOptions.consistent], vec: Point, /) -> Point:
    """Convert the point to a self-consistent set of units.

    Parameters
    ----------
    flag
        The point is converted to consistent units by looking for the first
        quantity with each physical type and converting all components to
        the units of that quantity.
    vec
        The point to convert.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.main as cx

    We can convert a point to the given units:

    >>> cart = cx.Point.from_([1, 2, 3], "m")

    If all you want is to convert to consistent units, you can use
    ``"consistent"``:

    >>> print(cart.uconvert(cx.ToUnitsOptions.consistent))
    <Point: chart=Cart3D (x, y, z) [m]
        [1 2 3]>

    >>> sph = cart.cconvert(cx.sph3d)
    >>> print(sph.uconvert(cx.ToUnitsOptions.consistent))
    <Point: chart=Spherical3D (r[m], theta[rad], phi[rad])
        [3.742 0.641 1.107]>

    """
    dim2unit = {}
    units_ = {}
    for k, v in vec.data.items():
        pt = u.dimension_of(v)
        if pt not in dim2unit:
            dim2unit[pt] = u.unit_of(v)
        units_[k] = dim2unit[pt]

    data = {k: u.uconvert(units_[k], v) for k, v in vec.data.items()}
    return replace(vec, data=data)


@plum.dispatch
def uconvert(usys: str, vec: Point, /) -> Point:
    """Convert the vector to the given units system.

    Parameters
    ----------
    usys
        The units system to convert to, as a string.
    vec
        The vector to convert.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.main as cx

    >>> usys = "galactic"
    >>> vector = cx.Point.from_([1, 2, 3], "m")
    >>> print(u.uconvert(usys, vector))
    <Point: chart=Cart3D (x, y, z) [kpc]
        [3.241e-20 6.482e-20 9.722e-20]>

    """
    usys: u.AbstractUnitSystem = u.unitsystem(usys)
    return cast("Point", u.uconvert(usys, vec))


# ============================================================================
# Type Conversion


@plum.conversion_method(Point, u.AbstractQuantity)
def point_to_q(obj: Point, /) -> u.AbstractQuantity:
    """`coordinax.Point` -> `unxt.Quantity`.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.main as cx
    >>> from plum import convert

    >>> vec = cx.Point.from_([1, 2, 3], "km")
    >>> convert(vec, u.AbstractQuantity)
    Q([1, 2, 3], 'km')

    >>> vec = cx.Point.from_(
    ...     {"r": u.Q(1, "km"), "theta": u.Q(2, "deg"), "phi": u.Q(3, "deg")},
    ...     cx.sph3d)
    >>> convert(vec, u.AbstractQuantity)
    QuantityMatrix([1, 2, 3], '(km, deg, deg)')

    >>> vec = cx.Point.from_(
    ...     {"rho": u.Q(1, "km"), "phi": u.Q(2, "deg"), "z": u.Q(3, "m")},
    ...     cx.cyl3d)
    >>> convert(vec, u.AbstractQuantity)
    QuantityMatrix([1, 2, 3], '(km, deg, m)')

    """
    # Pack the the data into value, unit tuple
    vals, units = pack_nonuniform_unit(obj.data, obj.chart.components)

    # Special case for homogeneous units (e.g. all angles in deg): we can
    # convert to a single unit and return a quantity with that unit.
    if len(set(units)) == 1:
        unit = u.unit("") if units[0] is None else units[0]
        return u.Q(vals, unit)

    return QuantityMatrix(vals, units)
