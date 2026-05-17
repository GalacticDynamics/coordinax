"""Astropy interoperability: QMatrix."""

__all__: tuple[str, ...] = ()

import numpy as np
import plum

import astropy.units as apyu

from coordinax.internal import QMatrix, UnitsMatrix


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


@plum.conversion_method(type_from=UnitsMatrix, type_to=apyu.StructuredUnit)
def unitsmatrix_to_structured_unit(obj: UnitsMatrix, /) -> apyu.StructuredUnit:
    """Convert a ``UnitsMatrix`` to an ``astropy.units.StructuredUnit``.

    >>> import plum
    >>> import astropy.units as apyu
    >>> from coordinax.internal import UnitsMatrix

    1D case:

    >>> umat = UnitsMatrix(("km", "s"))
    >>> plum.convert(umat, apyu.StructuredUnit)
    Unit("(km, s)")

    2D case:

    >>> umat2 = UnitsMatrix((("m", "s"), ("kg", "rad")))
    >>> plum.convert(umat2, apyu.StructuredUnit)
    Unit("((m, s), (kg, rad))")

    """
    return apyu.StructuredUnit(obj.to_tuple())


@plum.conversion_method(type_from=apyu.StructuredUnit, type_to=UnitsMatrix)
def structured_unit_to_unitsmatrix(obj: apyu.StructuredUnit, /) -> UnitsMatrix:
    """Convert an ``astropy.units.StructuredUnit`` to a ``UnitsMatrix``.

    >>> import plum
    >>> import astropy.units as apyu
    >>> from coordinax.internal import UnitsMatrix

    1D case:

    >>> su = apyu.StructuredUnit(("m", "s", "kg"))
    >>> result = plum.convert(su, UnitsMatrix)
    >>> result.shape
    (3,)
    >>> result[0]
    Unit("m")

    """
    return UnitsMatrix(_structured_unit_to_tuple(obj))


@plum.conversion_method(QMatrix, apyu.Quantity)
def convert_qmatrix_to_astropy_quantity(q: QMatrix, /) -> apyu.Quantity:
    """Convert a `coordinax.internal.QMatrix` to an `astropy.units.Quantity`.

    >>> import jax.numpy as jnp
    >>> import astropy.units as apyu
    >>> import plum
    >>> from coordinax.internal import QMatrix

    >>> qmat = QMatrix(jnp.array([1.0, 2.0]), unit=("km", "s"))
    >>> plum.convert(qmat, apyu.Quantity)
    <Quantity (1., 2.) (km, s)>

    """
    unit = plum.convert(q.unit, apyu.StructuredUnit)
    value_np = np.asarray(q.value)
    names = unit.field_names
    dtype = np.dtype([(name, value_np.dtype) for name in names])
    if value_np.ndim == 1:
        value_struct = np.array(tuple(value_np), dtype=dtype)
    else:
        value_struct = np.array([tuple(row) for row in value_np], dtype=dtype)
    return apyu.Quantity(value_struct, unit=unit)
