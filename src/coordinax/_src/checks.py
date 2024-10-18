"""Representation of coordinates in different systems."""

__all__: list[str] = []


import equinox as eqx

import quaxed.numpy as xp
from unxt import AbstractQuantity, Quantity, is_unit_convertible

from .typing import BatchableAngle, BatchableLength

_0m = Quantity(0, "meter")
_0d = Quantity(0, "rad")
_pid = Quantity(180, "deg")
_2pid = Quantity(360, "deg")


def check_r_non_negative(
    r: BatchableLength, /, _l: AbstractQuantity = _0m
) -> BatchableLength:
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


def check_azimuth_range(
    azimuth: BatchableAngle, /, _l: AbstractQuantity = _0d, _u: AbstractQuantity = _2pid
) -> BatchableAngle:
    """Check that the azimuthal angle is in the range.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from coordinax._src.checks import check_azimuth_range

    Pass through the input if it's in the range.

    >>> x = Quantity([0., 1, 2], "deg")
    >>> check_azimuth_range(x)
    Quantity['angle'](Array([0., 1., 2.], dtype=float32), unit='deg')

    Raise an error if anything is outside the range.

    >>> x = Quantity([0., 1, 2], "m")
    >>> try: check_azimuth_range(x)
    ... except Exception as e: print(e)
    The azimuthal angle must be in angular units...

    >>> x = Quantity([-1., 1, 2], "deg")
    >>> try: check_azimuth_range(x)
    ... except Exception: pass
    ... except Exception: pass

    """
    azimuth = eqx.error_if(
        azimuth,
        not is_unit_convertible("deg", azimuth),
        "The azimuthal angle must be in angular units.",
    )
    # TODO: enable integer support
    return eqx.error_if(
        azimuth,
        xp.any(xp.logical_or(xp.less(azimuth, _l), xp.greater_equal(azimuth, _u))),
        "The azimuthal angle must be in the range [0, 2pi).",
    )


def check_polar_range(
    polar: BatchableAngle, /, _l: AbstractQuantity = _0d, _u: AbstractQuantity = _pid
) -> BatchableAngle:
    """Check that the polar angle is in the range.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from coordinax._src.checks import check_polar_range

    Pass through the input if it's in the range.

    >>> x = Quantity([0., 1, 2], "deg")
    >>> check_polar_range(x)
    Quantity['angle'](Array([0., 1., 2.], dtype=float32), unit='deg')

    Raise an error if anything is outside thr range.

    >>> x = Quantity([0., 1, 2], "m")
    >>> try: check_polar_range(x)
    ... except Exception as e: print(e)
    The polar angle must be in angular units...

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
