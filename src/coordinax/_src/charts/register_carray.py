"""``carray`` dispatch implementations — pack component dicts into a QuantityMatrix."""

__all__ = ()

from typing import Final

import plum
import unxts.linalg as ul

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AllowValue

from coordinax._src.base import AbstractChart
from coordinax._src.custom_types import CDict

DMLS: Final = u.unit("")


@plum.dispatch
def carray(p: CDict, keys: tuple[str, ...], /) -> ul.QM:
    """Pack a component dict into a 1-D ``QM`` with per-component native units.

    Unitless components are treated as dimensionless, so quantity- and
    array-valued components can be packed together.

    >>> import unxt as u
    >>> import coordinax as cx
    >>> p = {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(3.0, "km")}
    >>> cx.carray(p, ("x", "y", "z"))
    QM([1., 2., 3.], '(km, km, km)')

    """
    units = tuple(u_ if (u_ := u.unit_of(p[k])) is not None else DMLS for k in keys)
    vals = [
        u.ustrip(AllowValue, unit, p[k]) for k, unit in zip(keys, units, strict=True)
    ]
    return ul.QM(jnp.stack(vals, axis=-1), unit=units)


@plum.dispatch
def carray(p: CDict, chart: AbstractChart, /) -> ul.QM:
    """Pack a component dict using ``chart.components`` as the keys.

    >>> import unxt as u
    >>> import coordinax as cx
    >>> p = {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km"), "z": u.Q(3.0, "km")}
    >>> cx.carray(p, cx.cart3d)
    QM([1., 2., 3.], '(km, km, km)')

    """
    return carray(p, chart.components)  # ty: ignore[missing-argument]


@plum.dispatch
def carray(p: CDict, keys: tuple[str, ...], usys: u.AbstractUnitSystem, /) -> ul.QM:
    """Pack a component dict, resolving each component's unit from ``usys``.

    >>> import unxt as u
    >>> import coordinax as cx
    >>> p = {"x": u.Q(1.0, "km"), "y": u.Q(2.0, "km")}
    >>> cx.carray(p, ("x", "y"), u.unitsystems.si)
    QM([1000., 2000.], '(m, m)')

    """
    units = tuple(
        usys[dim] if (dim := u.dimension_of(p[k])) is not None else DMLS for k in keys
    )
    vals = [
        u.ustrip(AllowValue, unit, p[k]) for k, unit in zip(keys, units, strict=True)
    ]
    return ul.QM(jnp.stack(vals, axis=-1), unit=units)


@plum.dispatch
def carray(p: CDict, keys: tuple[str, ...], unit: u.AbstractUnit, /) -> ul.QM:
    """Pack a component dict into a single shared ``unit`` (all components converted).

    >>> import unxt as u
    >>> import coordinax as cx
    >>> p = {"x": u.Q(1.0, "km"), "y": u.Q(200.0, "m")}
    >>> cx.carray(p, ("x", "y"), u.unit("km"))
    QM([1. , 0.2], '(km, km)')

    """
    vals = [u.ustrip(AllowValue, unit, p[k]) for k in keys]
    return ul.QM(jnp.stack(vals, axis=-1), unit=(unit,) * len(keys))
