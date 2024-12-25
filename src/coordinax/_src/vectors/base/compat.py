"""Intra-ecosystem Compatibility."""

__all__: list[str] = []


from collections.abc import Callable, Mapping
from dataclasses import Field
from typing import Any, Literal

from astropy.units import PhysicalType as Dimension
from jaxtyping import Shaped
from plum import conversion_method, convert, dispatch

import quaxed.numpy as jnp
import unxt as u
from dataclassish import field_values, replace
from unxt.quantity import AbstractQuantity, Quantity, UncheckedQuantity

from .attribute import VectorAttribute
from .base import AbstractVector
from .base_pos import AbstractPos
from .flags import AttrFilter
from .utils import ToUnitsOptions
from coordinax._src.distances import Distance
from coordinax._src.typing import Unit
from coordinax._src.vectors.utils import full_shaped

#####################################################################
# Convert to Quantity


@conversion_method(type_from=AbstractPos, type_to=AbstractQuantity)  # type: ignore[misc]
def convert_pos_to_absquantity(obj: AbstractPos, /) -> AbstractQuantity:
    """`coordinax.AbstractPos` -> `unxt.AbstractQuantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt.quantity import AbstractQuantity, Quantity

    >>> pos = cx.vecs.CartesianPos1D.from_([1.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1.], dtype=float32), unit='km')

    >>> pos = cx.vecs.RadialPos.from_([1.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1.], dtype=float32), unit='km')

    >>> pos = cx.vecs.CartesianPos2D.from_([1.0, 2.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 2.], dtype=float32), unit='km')

    >>> pos = cx.vecs.PolarPos(Quantity(1, "km"), Quantity(0, "deg"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 0.], dtype=float32, ...), unit='km')

    >>> pos = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='km')

    >>> pos = cx.SphericalPos(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "deg"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([0., 0., 1.], dtype=float32, ...), unit='km')

    >>> pos = cx.vecs.CylindricalPos(Quantity(1, "km"), Quantity(0, "deg"), Quantity(0, "km"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 0., 0.], dtype=float32, ...), unit='km')

    """  # noqa: E501
    cart = full_shaped(obj.vconvert(obj._cartesian_cls))  # noqa: SLF001
    return jnp.stack(tuple(field_values(cart)), axis=-1)


@conversion_method(type_from=AbstractPos, type_to=Quantity)  # type: ignore[misc]
def convert_pos_to_q(obj: AbstractPos, /) -> Quantity["length"]:
    """`coordinax.AbstractPos` -> `unxt.Quantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt.quantity import AbstractQuantity, Quantity

    >>> pos = cx.vecs.CartesianPos1D.from_([1], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1], dtype=int32), unit='km')

    >>> pos = cx.vecs.RadialPos.from_([1], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1], dtype=int32), unit='km')

    >>> pos = cx.vecs.CartesianPos2D.from_([1, 2], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1, 2], dtype=int32), unit='km')

    >>> pos = cx.vecs.PolarPos(Quantity(1, "km"), Quantity(0, "deg"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 0.], dtype=float32, ...), unit='km')

    >>> pos = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 2., 3.], dtype=float32), unit='km')

    >>> pos = cx.SphericalPos(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "deg"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([0., 0., 1.], dtype=float32, ...), unit='km')

    >>> pos = cx.vecs.CylindricalPos(Quantity(1, "km"), Quantity(0, "deg"), Quantity(0, "km"))
    >>> convert(pos, AbstractQuantity)
    Quantity['length'](Array([1., 0., 0.], dtype=float32, ...), unit='km')

    """  # noqa: E501
    return convert(convert(obj, AbstractQuantity), Quantity)


@conversion_method(type_from=AbstractPos, type_to=UncheckedQuantity)  # type: ignore[misc]
def convert_pos_to_uncheckedq(
    obj: AbstractPos, /
) -> Shaped[UncheckedQuantity, "*batch dims"]:
    """`coordinax.AbstractPos` -> `unxt.UncheckedQuantity`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt.quantity import AbstractQuantity, Quantity, UncheckedQuantity

    >>> pos = cx.vecs.CartesianPos1D.from_([1], "km")
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1], dtype=int32), unit='km')

    >>> pos = cx.vecs.RadialPos.from_([1], "km")
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1], dtype=int32), unit='km')

    >>> pos = cx.vecs.CartesianPos2D.from_([1, 2], "km")
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1, 2], dtype=int32), unit='km')

    >>> pos = cx.vecs.PolarPos(Quantity(1, "km"), Quantity(0, "deg"))
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1., 0.], dtype=float32, ...), unit='km')

    >>> pos = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1., 2., 3.], dtype=float32), unit='km')

    >>> pos = cx.SphericalPos(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "deg"))
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([0., 0., 1.], dtype=float32, ...), unit='km')

    >>> pos = cx.vecs.CylindricalPos(Quantity(1, "km"), Quantity(0, "deg"), Quantity(0, "km"))
    >>> convert(pos, UncheckedQuantity)
    UncheckedQuantity(Array([1., 0., 0.], dtype=float32, ...), unit='km')

    """  # noqa: E501
    return convert(convert(obj, AbstractQuantity), UncheckedQuantity)


@conversion_method(type_from=AbstractPos, type_to=Distance)  # type: ignore[misc]
def convert_pos_to_distance(obj: AbstractPos, /) -> Shaped[Distance, "*batch dims"]:
    """`coordinax.AbstractPos` -> `coordinax.Distance`.

    Examples
    --------
    >>> import coordinax as cx
    >>> from unxt.quantity import AbstractQuantity, Quantity
    >>> from coordinax.distance import Distance

    >>> pos = cx.vecs.CartesianPos1D.from_([1.0], "km")
    >>> convert(pos, Distance)
    Distance(Array([1.], dtype=float32), unit='km')

    >>> pos = cx.vecs.RadialPos.from_([1.0], "km")
    >>> convert(pos, Distance)
    Distance(Array([1.], dtype=float32), unit='km')

    >>> pos = cx.vecs.CartesianPos2D.from_([1, 2], "km")
    >>> convert(pos, Distance)
    Distance(Array([1, 2], dtype=int32), unit='km')

    >>> pos = cx.vecs.PolarPos(Quantity(1, "km"), Quantity(0, "deg"))
    >>> convert(pos, Distance)
    Distance(Array([1., 0.], dtype=float32, ...), unit='km')

    >>> pos = cx.CartesianPos3D.from_([1.0, 2.0, 3.0], "km")
    >>> convert(pos, Distance)
    Distance(Array([1., 2., 3.], dtype=float32), unit='km')

    >>> pos = cx.SphericalPos(Quantity(1.0, "km"), Quantity(0, "deg"), Quantity(0, "deg"))
    >>> convert(pos, Distance)
    Distance(Array([0., 0., 1.], dtype=float32, ...), unit='km')

    >>> pos = cx.vecs.CylindricalPos(Quantity(1, "km"), Quantity(0, "deg"), Quantity(0, "km"))
    >>> convert(pos, Distance)
    Distance(Array([1., 0., 0.], dtype=float32, ...), unit='km')

    """  # noqa: E501
    return convert(convert(obj, AbstractQuantity), Distance)


#####################################################################
# Compatibility with unxt


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

    >>> cart = cx.vecs.CartesianPos2D(x=Quantity(1, "m"), y=Quantity(2, "km"))
    >>> cart.uconvert({u.dimension("length"): "km"})
    CartesianPos2D(
        x=Quantity[...](value=...f32[], unit=Unit("km")),
        y=Quantity[...](value=...i32[], unit=Unit("km"))
    )

    This also works for vectors with different units:

    >>> sph = cx.SphericalPos(r=Quantity(1, "m"), theta=Quantity(45, "deg"),
    ...                       phi=Quantity(3, "rad"))
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

    >>> cart = cx.vecs.CartesianPos2D(x=Quantity(1, "m"), y=Quantity(2, "km"))
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

    >>> sph = cx.SphericalPos(r=Quantity(1, "m"), theta=Quantity(45, "deg"),
    ...                       phi=Quantity(3, "rad"))
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


#####################################################################
# Compatibility with dataclassish


def _is_attr(obj_cls: type[AbstractVector], name: str) -> bool:
    """Check if the attribute is a `VectorAttribute`."""
    return isinstance(getattr(obj_cls, name, None), VectorAttribute)


# -------------------------------------------------------------------


@dispatch
def fields(flag: type[AttrFilter], obj: object) -> tuple[Field, ...]:  # type: ignore[type-arg]
    """Return the fields of an object, filtering out `VectorAttribute`s.

    Examples
    --------
    An example vector class with a `VectorAttribute` field is `FourVector`.

    >>> from dataclassish import fields
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> [f.name for f in fields(cx.vecs.AttrFilter, w)]
    ['t', 'q']

    """
    return fields(flag, type(obj))


@dispatch
def fields(flag: type[AttrFilter], obj_cls: type[object]) -> tuple[Field, ...]:  # type: ignore[type-arg]
    """Return the fields of an object, filtering out `VectorAttribute`s.

    Examples
    --------
    An example vector class with a `VectorAttribute` field is `FourVector`.

    >>> import dataclasses
    >>> import dataclassish
    >>> import coordinax as cx

    >>> [f.name for f in dataclasses.fields(cx.FourVector)]
    ['t', 'q', 'c']

    >>> [f.name for f in dataclassish.fields(cx.vecs.AttrFilter, cx.FourVector)]
    ['t', 'q']

    """
    return tuple(
        field for field in fields(obj_cls) if not _is_attr(obj_cls, field.name)
    )


# -------------------------------------------------------------------


@dispatch  # type: ignore[misc]
def asdict(
    flag: type[AttrFilter],
    obj: AbstractVector,
    /,
    *,
    dict_factory: Callable[[list[tuple[str, Any]]], dict[str, Any]] = dict,
) -> dict[str, Any]:
    """Return object, filtering out `VectorAttribute`s, fields as a dict.

    Examples
    --------
    An example vector class with a `VectorAttribute` field is `FourVector`.

    >>> from dataclassish import asdict
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> asdict(cx.vecs.AttrFilter, w).keys()
    dict_keys(['t', 'q'])

    """
    obj_cls = type(obj)
    return dict_factory(
        [(k, v) for k, v in asdict(obj).items() if not _is_attr(obj_cls, k)]
    )


# -------------------------------------------------------------------


@dispatch
def field_keys(flag: type[AttrFilter], obj: AbstractVector, /) -> tuple[str, ...]:
    """Return field keys, filtering out `VectorAttribute`s.

    Examples
    --------
    An example vector class with a `VectorAttribute` field is `FourVector`.

    >>> from dataclassish import field_keys
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> field_keys(cx.vecs.AttrFilter, w)
    ('t', 'q')

    """
    return tuple(f.name for f in fields(flag, obj))


@dispatch
def field_keys(
    flag: type[AttrFilter], obj_cls: type[AbstractVector], /
) -> tuple[str, ...]:
    """Return field keys, filtering out `VectorAttribute`s.

    Examples
    --------
    An example vector class with a `VectorAttribute` field is `FourVector`.

    >>> from dataclassish import field_keys
    >>> import coordinax as cx

    >>> field_keys(cx.vecs.AttrFilter, cx.FourVector)
    ('t', 'q')

    """
    return tuple(f.name for f in fields(flag, obj_cls))


# -------------------------------------------------------------------


@dispatch  # type: ignore[misc]
def field_values(flag: type[AttrFilter], obj: AbstractVector, /) -> tuple[Any, ...]:
    """Return field values, filtering out `VectorAttribute`s.

    Examples
    --------
    An example vector class with a `VectorAttribute` field is `FourVector`.

    >>> from dataclassish import asdict
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> field_keys(cx.vecs.AttrFilter, w)
    ('t', 'q')

    """
    obj_cls = type(obj)
    return tuple(v for k, v in field_items(obj) if not _is_attr(obj_cls, k))


# -------------------------------------------------------------------


@dispatch  # type: ignore[misc]
def field_items(
    flag: type[AttrFilter], obj: AbstractVector, /
) -> tuple[tuple[str, Any], ...]:
    """Return field items, filtering out `VectorAttribute`s.

    Examples
    --------
    An example vector class with a `VectorAttribute` field is `FourVector`.

    >>> from dataclassish import field_items
    >>> import unxt as u
    >>> import coordinax as cx

    >>> w = cx.FourVector(t=u.Quantity(1, "s"), q=u.Quantity([1, 2, 3], "m"))
    >>> [(f[0], type(f[1]).__name__) for f in field_items(cx.vecs.AttrFilter, w)]
    [('t', "Quantity[PhysicalType('time')]"), ('q', 'CartesianPos3D')]

    """
    obj_cls = type(obj)
    return tuple((k, v) for k, v in field_items(obj) if not _is_attr(obj_cls, k))
