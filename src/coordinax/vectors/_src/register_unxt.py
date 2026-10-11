"""Point."""

__all__ = ("ToUnitsOptions",)

from dataclasses import replace
from enum import Enum

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, cast, final

import plum

import unxt as u
import unxts.linalg as ul

import coordinaxs.api.charts as cxcapi
from .bundle import Coordinate
from .point import Point
from .tangent import Tangent

if TYPE_CHECKING:
    import unxts.linalg as ul


@final
class ToUnitsOptions(Enum):
    """Options for the units argument of `uconvert`.

    This enum provides named conversion behaviors that are accepted by
    `Point.uconvert`.

    Examples
    --------
    `Point.uconvert` with `consistent`:

    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.Point.from_({"x": u.Q(1, "m"), "y": u.Q(2, "km")}, cx.cart2d)
    >>> print(vec.uconvert(cx.ToUnitsOptions.consistent))
    <Point: chart=Cart2D (x, y) [m]
        [1.e+00 2.e+03]>

    """

    consistent = "consistent"
    """Convert to consistent units."""


# ============================================================================
# `Point` and `Tangent`: unit conversion is per component, so one set of rules
# serves both. Each component's target unit comes from its *own* dimension, not
# the chart's `coord_dimensions` -- a velocity's ``x`` is length/time.


@plum.dispatch
def uconvert(usys: u.AbstractUnitSystem, vec: Point | Tangent, /) -> Point | Tangent:
    """Convert the vector to the given unit system.

    Each component is converted to the unit system's unit for that component's
    own dimension.

    >>> import unxt as u
    >>> import coordinax as cx

    >>> usys = u.unitsystem("m", "s", "kg", "rad")

    >>> vec = cx.Point.from_([1, 2, 3], "km")
    >>> print(u.uconvert(usys, vec))
    <Point: chart=Cart3D (x, y, z) [m]
        [1000. 2000. 3000.]>

    A tangent's components carry their own dimension (here speed), which the
    unit system resolves:

    >>> vel = cx.Tangent.from_(u.Q([1., 2, 3], "km/s"), cx.cart3d)
    >>> print(u.uconvert(usys, vel))
    <Tangent: chart=Cart3D (x, y, z) [m / s]
        [1000. 2000. 3000.]>

    """
    data = {k: u.uconvert(usys[u.dimension_of(v)], v) for k, v in vec.data.items()}
    return replace(vec, data=data)


@plum.dispatch
def uconvert(
    units: Mapping[u.AbstractDimension, u.AbstractUnit | str], vec: Point | Tangent, /
) -> Point | Tangent:
    """Convert the vector to the given units, by dimension.

    Components whose dimension is not in ``units`` keep their unit.

    >>> import unxt as u
    >>> import coordinax as cx

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

    And for tangents, keyed by the components' own dimension:

    >>> vel = cx.Tangent.from_(u.Q([1., 2, 3], "km/s"), cx.cart3d)
    >>> print(vel.uconvert({u.dimension("speed"): "m/s"}))
    <Tangent: chart=Cart3D (x, y, z) [m / s]
        [1000. 2000. 3000.]>

    """
    units_ = {u.dimension(k): v for k, v in units.items()}
    data = {
        k: u.uconvert(units_.get(u.dimension_of(v), u.unit_of(v)), v)
        for k, v in vec.data.items()
    }
    return replace(vec, data=data)


@plum.dispatch
def uconvert(units: Mapping[str, Any], vec: Point | Tangent, /) -> Point | Tangent:
    """Convert the vector to the given units, by component name.

    >>> import unxt as u
    >>> import coordinax as cx

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

    And for tangents:

    >>> vel = cx.Tangent.from_(u.Q([1., 2, 3], "km/s"), cx.cart3d)
    >>> print(vel.uconvert({"x": "m/s", "y": "m/s", "z": "m/s"}))
    <Tangent: chart=Cart3D (x, y, z) [m / s]
        [1000. 2000. 3000.]>

    """
    data = {  # (component: unit)
        k: u.uconvert(units.get(k, u.unit_of(v)), v)  # default to original unit
        for k, v in vec.data.items()
    }
    return replace(vec, data=data)


@plum.dispatch
def uconvert(
    flag: Literal[ToUnitsOptions.consistent], vec: Point | Tangent, /
) -> Point | Tangent:
    """Convert the vector to a self-consistent set of units.

    The vector is converted to consistent units by looking for the first
    quantity with each physical type and converting all components to the units
    of that quantity.

    >>> import unxt as u
    >>> import coordinax as cx

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

    >>> vel = cx.Tangent.from_(
    ...     {"x": u.Q(1., "km/s"), "y": u.Q(2., "m/s")}, cx.cart2d)
    >>> print(vel.uconvert(cx.ToUnitsOptions.consistent))
    <Tangent: chart=Cart2D (x, y) [km / s]
        [1.    0.002]>

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
def uconvert(usys: str, vec: Point | Tangent, /) -> Point | Tangent:
    """Convert the vector to the given (named) unit system.

    >>> import unxt as u
    >>> import coordinax as cx

    >>> usys = "galactic"
    >>> vector = cx.Point.from_([1, 2, 3], "m")
    >>> print(u.uconvert(usys, vector))
    <Point: chart=Cart3D (x, y, z) [kpc]
        [3.241e-20 6.482e-20 9.722e-20]>

    >>> vel = cx.Tangent.from_(u.Q([1., 2, 3], "km/s"), cx.cart3d)
    >>> print(u.uconvert(usys, vel))
    <Tangent: chart=Cart3D (x, y, z) [kpc / Myr]
        [0.001 0.002 0.003]>

    """
    usys_: u.AbstractUnitSystem = u.unitsystem(usys)
    return cast("Point | Tangent", u.uconvert(usys_, vec))


# ============================================================================
# `Coordinate`: convert the base point and every fibre.


@plum.dispatch
def uconvert(
    units: u.AbstractUnitSystem | str | Mapping | ToUnitsOptions, vec: Coordinate, /
) -> Coordinate:
    """Convert a coordinate's base point and each of its fibres.

    ``units`` is applied to every member independently, so a unit system, a
    dimension-keyed mapping, or `ToUnitsOptions.consistent` converts the point
    and the fibres alike. In a dimension-keyed mapping each member takes only
    the entries for its own dimensions.

    >>> import unxt as u
    >>> import coordinax as cx

    >>> pv = cx.Coordinate(
    ...     point=cx.Point.from_([1., 2, 3], "kpc"),
    ...     velocity=cx.Tangent.from_(u.Q([4., 5, 6], "km/s"), cx.cart3d))

    >>> out = u.uconvert({u.dimension("length"): "pc",
    ...                   u.dimension("speed"): "m/s"}, pv)
    >>> print(out.point)
    <Point: chart=Cart3D (x, y, z) [pc]
        [1000. 2000. 3000.]>
    >>> print(out["velocity"])
    <Tangent: chart=Cart3D (x, y, z) [m / s]
        [4000. 5000. 6000.]>

    >>> print(pv.uconvert("galactic")["velocity"])
    <Tangent: chart=Cart3D (x, y, z) [kpc / Myr]
        [0.004 0.005 0.006]>

    """
    return Coordinate._create_unchecked(
        cast("Point", u.uconvert(units, vec.point)),
        {k: cast("Tangent", u.uconvert(units, v)) for k, v in vec.items()},
    )


@plum.dispatch
def uconvert(units: Mapping[str, Any], vec: Coordinate, /) -> Coordinate:
    """Convert a coordinate's members by component name.

    A component name such as ``"x"`` is shared by the point and its fibres,
    but they differ in dimension (length vs. length/time). So each entry is
    applied to a member only where its unit has that component's dimension;
    elsewhere the component keeps its unit. Convert a single member through
    its own `uconvert` when that is not what you want.

    >>> import unxt as u
    >>> import coordinax as cx

    >>> pv = cx.Coordinate(
    ...     point=cx.Point.from_([1., 2, 3], "kpc"),
    ...     velocity=cx.Tangent.from_(u.Q([4., 5, 6], "km/s"), cx.cart3d))

    ``"pc"`` matches the point's ``x``; ``"m/s"`` matches the velocity's ``y``:

    >>> out = pv.uconvert({"x": "pc", "y": "m/s"})
    >>> print(out.point)
    <Point: chart=Cart3D (x[pc], y[kpc], z[kpc])
        [1000.    2.    3.]>
    >>> print(out["velocity"])
    <Tangent: chart=Cart3D (x[km / s], y[m / s], z[km / s])
        [4.e+00 5.e+03 6.e+00]>

    """

    def _convert(member: Any, /) -> Any:
        matching = {
            k: unit
            for k, unit in units.items()
            if k in member.data
            and u.dimension_of(u.unit(unit)) == u.dimension_of(member.data[k])
        }
        # An empty dict would be ambiguous between the keyed-mapping rules.
        return u.uconvert(matching, member) if matching else member

    return Coordinate._create_unchecked(
        _convert(vec.point), {k: _convert(v) for k, v in vec.items()}
    )


# ============================================================================
# Type Conversion


@plum.conversion_method(Point, u.AbstractQuantity)
def point_to_q(obj: Point, /) -> u.AbstractQuantity:
    """`coordinax.Point` -> `unxt.Quantity`.

    >>> import unxt as u
    >>> import coordinax as cx
    >>> from plum import convert

    >>> vec = cx.Point.from_([1, 2, 3], "km")
    >>> convert(vec, u.AbstractQuantity)
    Q([1, 2, 3], 'km')

    >>> vec = cx.Point.from_(
    ...     {"r": u.Q(1, "km"), "theta": u.Q(2, "deg"), "phi": u.Q(3, "deg")},
    ...     cx.sph3d)
    >>> convert(vec, u.AbstractQuantity)
    QM([1, 2, 3], '(km, deg, deg)')

    >>> vec = cx.Point.from_(
    ...     {"rho": u.Q(1, "km"), "phi": u.Q(2, "deg"), "z": u.Q(3, "m")},
    ...     cx.cyl3d)
    >>> convert(vec, u.AbstractQuantity)
    QM([1, 2, 3], '(km, deg, m)')

    """
    # Pack the data into value, unit tuple
    qm = cast("ul.QM", cxcapi.carray(obj.data, obj.chart.components))
    vals, units = qm.value, qm.unit.to_tuple()

    # Special case for homogeneous units (e.g. all angles in deg): we can
    # convert to a single unit and return a quantity with that unit.
    if len(set(units)) == 1:
        unit = u.unit("") if units[0] is None else units[0]
        return u.Q(vals, unit)

    return qm
