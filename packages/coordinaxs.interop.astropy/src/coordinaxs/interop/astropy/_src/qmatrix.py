"""Astropy interoperability: QuantityMatrix."""

__all__: tuple[str, ...] = ()

from typing import Any

import astropy.units as apyu
import numpy as np
import plum

import unxts.linalg as ul


def _structured_unit_to_tuple(obj: apyu.StructuredUnit) -> tuple:
    """Recursively convert an astropy ``StructuredUnit`` to a nested tuple of units.

    Handles both 1-D (flat structured) and 2-D (nested structured) unit layouts.
    Leaf values are astropy unit objects (subclasses of ``u.AbstractUnit``).
    """
    result = []
    for v in obj.values():
        if isinstance(v, apyu.StructuredUnit):
            result.append(_structured_unit_to_tuple(v))
        else:
            result.append(v)  # leaf: astropy.units.UnitBase → u.AbstractUnit
    return tuple(result)


def _structured_depth(unit: apyu.StructuredUnit) -> int:
    """How many axes of a value the layout of *unit* describes."""
    sub = next(iter(unit.values()))
    return 1 + _structured_depth(sub) if isinstance(sub, apyu.StructuredUnit) else 1


def _structured_dtype(unit: apyu.StructuredUnit, base: np.dtype) -> np.dtype:
    """Build the numpy dtype for *unit*'s layout, with *base* at every leaf."""
    return np.dtype(
        [
            (
                name,
                _structured_dtype(sub, base)
                if isinstance(sub, apyu.StructuredUnit)
                else base,
            )
            for name, sub in unit.items()
        ]
    )


def _records(value: np.ndarray) -> Any:
    """Group *value* into nested tuples -- how numpy spells a structured scalar."""
    return tuple(_records(v) for v in value) if value.ndim else value[()]


@plum.conversion_method(type_from=ul.UnitsMatrix, type_to=apyu.StructuredUnit)
def unitsmatrix_to_structured_unit(obj: ul.UnitsMatrix, /) -> apyu.StructuredUnit:
    """Convert a ``UnitsMatrix`` to an ``astropy.units.StructuredUnit``.

    >>> import plum
    >>> import astropy.units as apyu
    >>> import unxts.linalg as ul

    1D case:

    >>> umat = ul.UnitsMatrix(("km", "s"))
    >>> plum.convert(umat, apyu.StructuredUnit)
    Unit("(km, s)")

    2D case:

    >>> umat2 = ul.UnitsMatrix((("m", "s"), ("kg", "rad")))
    >>> plum.convert(umat2, apyu.StructuredUnit)
    Unit("((m, s), (kg, rad))")

    """
    return apyu.StructuredUnit(obj.to_tuple())


@plum.conversion_method(type_from=apyu.StructuredUnit, type_to=ul.UnitsMatrix)
def structured_unit_to_unitsmatrix(obj: apyu.StructuredUnit, /) -> ul.UnitsMatrix:
    """Convert an ``astropy.units.StructuredUnit`` to a ``UnitsMatrix``.

    >>> import plum
    >>> import astropy.units as apyu
    >>> import unxts.linalg as ul

    1D case:

    >>> su = apyu.StructuredUnit(("m", "s", "kg"))
    >>> result = plum.convert(su, ul.UnitsMatrix)
    >>> result.shape
    (3,)
    >>> result[0]
    Unit("m")

    """
    return ul.UnitsMatrix(_structured_unit_to_tuple(obj))


@plum.conversion_method(ul.QuantityMatrix, apyu.Quantity)
def convert_quantitymatrix_to_astropy_quantity(
    q: ul.QuantityMatrix, /
) -> apyu.Quantity:
    """Convert a `unxts.linalg.QuantityMatrix` to an `astropy.units.Quantity`.

    >>> import jax.numpy as jnp
    >>> import astropy.units as apyu
    >>> import plum
    >>> import unxts.linalg as ul

    >>> qmat = ul.QuantityMatrix(jnp.array([1.0, 2.0]), unit=("km", "s"))
    >>> plum.convert(qmat, apyu.Quantity)
    <Quantity (1., 2.) (km, s)>

    A nested unit layout nests the record to match:

    >>> qmat = ul.QuantityMatrix(jnp.array([[1.0, 2.0], [3.0, 4.0]]),
    ...                          unit=(("m", "s"), ("kg", "rad")))
    >>> plum.convert(qmat, apyu.Quantity)
    <Quantity ((1., 2.), (3., 4.)) ((m, s), (kg, rad))>

    A flat layout instead reads one leading axis as batch:

    >>> qmat = ul.QuantityMatrix(jnp.arange(6.0).reshape(2, 3),
    ...                          unit=("m", "s", "kg"))
    >>> plum.convert(qmat, apyu.Quantity)
    <Quantity [(0., 1., 2.), (3., 4., 5.)] (m, s, kg)>

    """
    unit = plum.convert(q.unit, apyu.StructuredUnit)
    value = np.asarray(q.value)

    # The unit layout claims the trailing axes -- one per level of nesting --
    # and whatever is left in front of them is batch. A flat layout over a
    # (4, 3) value is four records; a nested one over a (2, 2) value is a
    # single record of records.
    depth = _structured_depth(unit)
    dtype = _structured_dtype(unit, value.dtype)
    if value.ndim == depth:
        data = _records(value)
    elif value.ndim == depth + 1:
        data = [_records(row) for row in value]
    else:
        msg = (
            f"cannot lay a value of shape {value.shape} out under the unit "
            f"{unit}: the layout claims the last {depth} of those axes and "
            f"astropy allows one batch axis in front of them, not "
            f"{value.ndim - depth}."
        )
        raise ValueError(msg)

    return apyu.Quantity(np.array(data, dtype=dtype), unit=unit)
