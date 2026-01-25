"""Vector."""

__all__ = ("ToUnitsOptions",)

from enum import Enum

from collections.abc import Mapping
from typing import Any, Literal, final

import plum

import unxt as u

from .base import Vector


@final
class ToUnitsOptions(Enum):
    """Options for the units argument of `Vector.uconvert`."""

    consistent = "consistent"
    """Convert to consistent units."""


@plum.dispatch
def uconvert(usys: u.AbstractUnitSystem, vec: Vector, /) -> Vector:
    """Convert the vector to the given units.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    >>> usys = u.unitsystem("m", "s", "kg", "rad")

    >>> vec = cx.Vector.from_([1, 2, 3], "km")
    >>> print(u.uconvert(usys, vec))
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [1000. 2000. 3000.]>

    """
    data = {k: u.uconvert(usys[u.dimension_of(v)], v) for k, v in vec.data.items()}
    return Vector(data, chart=vec.chart, role=vec.role)


@plum.dispatch
def uconvert(
    units: Mapping[u.AbstractDimension, u.AbstractUnit | str], vec: Vector, /
) -> Vector:
    """Convert the vector to the given units.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    We can convert a vector to the given units:

    >>> cart = cx.Vector.from_({"x": u.Q(1, "m"), "y": u.Q(2, "km")})
    >>> print(cart.uconvert({u.dimension("length"): "km"}))
    <Vector: chart=Cart2D, role=Point (x, y) [km]
        [1.e-03 2.e+00]>

    This also works for vectors with different units:

    >>> sph = cx.Vector(
    ...     data={"r": u.Q(1, "m"), "theta": u.Q(45, "deg"), "phi": u.Q(3, "rad")},
    ...     chart=cx.charts.sph3d, role=cx.roles.Point())
    >>> print(sph.uconvert({u.dimension("length"): "km", u.dimension("angle"): "deg"}))
    <Vector: chart=Spherical3D, role=Point (r[km], theta[deg], phi[deg])
        [1.000e-03 4.500e+01 1.719e+02]>

    """
    # # Ensure `units_` is PT -> Unit
    units_ = {u.dimension(k): v for k, v in units.items()}
    data = {k: u.uconvert(units_[u.dimension_of(v)], v) for k, v in vec.data.items()}
    return Vector(data, chart=vec.chart, role=vec.role)


@plum.dispatch
def uconvert(units: Mapping[str, Any], vec: Vector, /) -> Vector:
    """Convert the vector to the given units.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    We can convert a vector to the given units:

    >>> cart = cx.Vector.from_({"x": u.Q(1, "m"), "y": u.Q(2, "km")})
    >>> print(cart.uconvert({"x": "km", "y": "m"}))
    <Vector: chart=Cart2D, role=Point (x[km], y[m])
        [1.e-03 2.e+03]>

    This also works for converting just some of the components:

    >>> print(cart.uconvert({"x": "km"}))
    <Vector: chart=Cart2D, role=Point (x, y) [km]
        [1.e-03 2.e+00]>

    This also works for vectors with different units:

    >>> sph = cx.Vector(
    ...     data={"r": u.Q(1, "m"), "theta": u.Q(45, "deg"), "phi": u.Q(3, "rad")},
    ...     chart=cx.charts.sph3d, role=cx.roles.Point())
    >>> print(sph.uconvert({"r": "km", "theta": "rad"}))
    <Vector: chart=Spherical3D, role=Point (r[km], theta[rad], phi[rad])
        [1.000e-03 7.854e-01 3.000e+00]>

    """
    data = {  # (component: unit)
        k: u.uconvert(units.get(k, u.unit_of(v)), v)  # default to original unit
        for k, v in vec.data.items()
    }
    return Vector(data, chart=vec.chart, role=vec.role)


@plum.dispatch
def uconvert(flag: Literal[ToUnitsOptions.consistent], vec: Vector, /) -> Vector:
    """Convert the vector to a self-consistent set of units.

    Parameters
    ----------
    flag
        The vector is converted to consistent units by looking for the first
        quantity with each physical type and converting all components to
        the units of that quantity.
    vec
        The vector to convert.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax as cx

    We can convert a vector to the given units:

    >>> cart = cx.Vector.from_([1, 2, 3], "m")

    If all you want is to convert to consistent units, you can use
    ``"consistent"``:

    >>> print(cart.uconvert(cx.objs.ToUnitsOptions.consistent))
    <Vector: chart=Cart3D, role=Point (x, y, z) [m]
        [1 2 3]>

    >>> sph = cart.vconvert(cx.charts.sph3d)
    >>> print(sph.uconvert(cx.objs.ToUnitsOptions.consistent))
    <Vector: chart=Spherical3D, role=Point (r[m], theta[rad], phi[rad])
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
    return Vector(data, chart=vec.chart, role=vec.role)


@plum.dispatch
def uconvert(usys: str, vec: Vector, /) -> Vector:
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
    >>> import coordinax as cx

    >>> usys = "galactic"
    >>> vector = cx.Vector.from_([1, 2, 3], "m")
    >>> print(u.uconvert(usys, vector))
    <Vector: chart=Cart3D, role=Point (x, y, z) [kpc]
        [3.241e-20 6.482e-20 9.722e-20]>

    """
    usys = u.unitsystem(usys)
    return uconvert(usys, vec)
