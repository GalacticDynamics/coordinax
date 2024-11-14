"""Representation of coordinates in different systems."""

__all__: list[str] = []


import equinox as eqx

import quaxed.numpy as xp
from unxt import is_unit_convertible
from unxt.quantity import AbstractQuantity

from .angle import Angle
from .distance import Distance
from .typing import BatchableAngle, BatchableAngleQ

_0m = Distance(0, "meter")
_0d = Angle(0, "rad")
_pid = Angle(180, "deg")
_2pid = Angle(360, "deg")


def check_r_non_negative(
    r: AbstractQuantity, /, _l: Distance = _0m
) -> AbstractQuantity:
    """Check that the radial distance is non-negative.

    Examples
    --------
    >>> from unxt import Quantity

    Pass through the input if the radial distance is non-negative.

    >>> x = Quantity([0, 1, 2], "m")
    >>> check_r_non_negative(x)
    Quantity['length'](Array([0, 1, 2], dtype=int32), unit='m')

    Raise an error if the radial distance is negative.

    >>> x = Quantity([-1, 1, 2], "m")
    >>> try: check_r_non_negative(x)
    ... except Exception: pass

    """
    return eqx.error_if(r, xp.any(r < _l), "The radial distance must be non-negative.")


def check_polar_range(
    polar: BatchableAngle | BatchableAngleQ, /, _l: Angle = _0d, _u: Angle = _pid
) -> BatchableAngle | BatchableAngleQ:
    """Check that the polar angle is in the range.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from coordinax._src.checks import check_polar_range

    Pass through the input if it's in the range.

    >>> x = Quantity([0., 1, 2], "deg")
    >>> check_polar_range(x)
    Quantity['angle'](Array([0., 1., 2.], dtype=float32), unit='deg')

    Raise an error if anything is outside the range.

    >>> x = Quantity([0., 1, 2], "m")
    >>> try: check_polar_range(x)
    ... except Exception as e: print("wrong units")
    wrong units

    >>> x = Quantity([-1., 1, 2], "deg")
    >>> try: check_polar_range(x)
    ... except Exception: pass

    """
    polar = eqx.error_if(
        polar,
        not is_unit_convertible("deg", polar),
        "The polar angle must be in angular units.",
    )
    return eqx.error_if(
        polar,
        xp.any(xp.logical_or((polar < _l), (polar > _u))),
        "The inclination angle must be in the range [0, pi].",
    )
