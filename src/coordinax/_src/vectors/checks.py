"""Representation of coordinates in different systems."""

__all__: list[str] = []


import equinox as eqx

import quaxed.numpy as jnp
import unxt as u
from unxt.quantity import AbstractQuantity

from coordinax._src.angles import Angle, BatchableAngleQ

_0d = Angle(0, "rad")
_pid = Angle(180, "deg")


def check_r_non_negative(r: AbstractQuantity, /) -> AbstractQuantity:
    """Check that the radial distance is non-negative.

    Examples
    --------
    >>> import unxt as u

    Pass through the input if the radial distance is non-negative.

    >>> x = u.Quantity([0, 1, 2], "m")
    >>> check_r_non_negative(x)
    Quantity['length'](Array([0, 1, 2], dtype=int32), unit='m')

    Raise an error if the radial distance is negative.

    >>> x = u.Quantity([-1, 1, 2], "m")
    >>> try: check_r_non_negative(x)
    ... except Exception: pass

    """
    return check_non_negative(r, name="radial distance r")


def check_polar_range(
    polar: BatchableAngleQ,
    /,
    _l: Angle = _0d,
    _u: Angle = _pid,
) -> BatchableAngleQ:
    """Check that the polar angle is in the range.

    Examples
    --------
    >>> import unxt as u
    >>> from coordinax._src.vectors.checks import check_polar_range

    Pass through the input if it's in the range.

    >>> x = u.Quantity([0., 1, 2], "deg")
    >>> check_polar_range(x)
    Quantity['angle'](Array([0., 1., 2.], dtype=float32), unit='deg')

    Raise an error if anything is outside the range.

    >>> x = u.Quantity([0., 1, 2], "m")
    >>> try: check_polar_range(x)
    ... except Exception as e: print("wrong units")
    wrong units

    >>> x = u.Quantity([-1., 1, 2], "deg")
    >>> try: check_polar_range(x)
    ... except Exception: pass

    """
    polar = eqx.error_if(
        polar,
        not u.is_unit_convertible("deg", polar),
        "The polar angle must be in angular units.",
    )
    return eqx.error_if(
        polar,
        jnp.any(jnp.logical_or((polar < _l), (polar > _u))),
        "The inclination angle must be in the range [0, pi].",
    )


def check_non_negative(x: AbstractQuantity, /, *, name: str = "") -> AbstractQuantity:
    """Check that the input is non-negative.

    Examples
    --------
    >>> import unxt as u

    Pass through the input if the value is non-negative.

    >>> x = u.Quantity([0, 1, 2], "m")
    >>> check_non_negative(x)
    Quantity['length'](Array([0, 1, 2], dtype=int32), unit='m')

    Raise an error if any value is negative.

    >>> x = u.Quantity([-1, 1, 2], "m")
    >>> try: check_non_negative(x)
    ... except Exception: pass

    """
    name = f" {name}" if name else name
    return eqx.error_if(x, jnp.any(x < 0), f"The input{name} must be non-negative.")


def check_non_negative_non_zero(
    x: AbstractQuantity, /, *, name: str = ""
) -> AbstractQuantity:
    """Check that the input is non-negative and non-zero.

    Examples
    --------
    >>> import unxt as u

    Pass through the input if the value is non-negative.

    >>> x = u.Quantity([1, 2, 3], "m")
    >>> check_non_negative_non_zero(x)
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    Raise an error if any value is negative or zero.

    >>> x = u.Quantity([-1, 1, 2], "m")
    >>> try: check_non_negative_non_zero(x)
    ... except Exception: pass

    >>> x = u.Quantity([0, 1, 2], "m")
    >>> try: check_non_negative_non_zero(x)
    ... except Exception: pass

    """
    name = f" {name}" if name else name
    return eqx.error_if(
        x, jnp.any(x <= 0), f"The input{name} must be non-negative and non-zero."
    )


def check_less_than(
    x: AbstractQuantity,
    max_val: AbstractQuantity,
    /,
    *,
    name: str = "",
    comparison_name: str = "the specified maximum value",
) -> AbstractQuantity:
    """Check that the input value is less than the input maximum value.

    Examples
    --------
    >>> import unxt as u

    Pass through the input if the value is less than the max value:

    >>> x = u.Quantity([1, 2, 3], "m")
    >>> check_less_than(x, u.Quantity(4, "m"))
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    Raise an error if the input is larger than the maximum value.

    >>> try: check_less_than(x, u.Quantity(1.5, "m"))
    ... except Exception: pass

    """
    name = f" {name}" if name else name
    msg = f"The input{name} must be less than {comparison_name}."
    return eqx.error_if(x, jnp.any(x >= max_val), msg)


def check_less_than_equal(
    x: AbstractQuantity,
    max_val: AbstractQuantity,
    /,
    *,
    name: str = "",
    comparison_name: str = "the specified maximum value",
) -> AbstractQuantity:
    """Check that the input value is less than or equal to the input maximum value.

    Examples
    --------
    >>> import unxt as u

    Pass through the input if the value is less than or equal to the max value:

    >>> x = u.Quantity([1, 2, 3], "m")
    >>> check_less_than_equal(x, u.Quantity(3, "m"))
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    Raise an error if the input is larger than the maximum value.

    >>> try: check_less_than_equal(x, u.Quantity(2, "m"))
    ... except Exception: pass

    """
    name = f" {name}" if name else name
    msg = f"The input{name} must be less than or equal to {comparison_name}."
    return eqx.error_if(x, jnp.any(x > max_val), msg)


def check_greater_than(
    x: AbstractQuantity,
    min_val: AbstractQuantity,
    /,
    *,
    name: str = "",
    comparison_name: str = "the specified minimum value",
) -> AbstractQuantity:
    """Check that the input value is greater than the input minimum value.

    Examples
    --------
    >>> import unxt as u

    Pass through the input if the value is greater than the min value:

    >>> x = u.Quantity([1, 2, 3], "m")
    >>> check_greater_than(x, u.Quantity(0, "m"))
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    Raise an error if the input is smaller than the minimum value.

    >>> try: check_greater_than(x, u.Quantity(4, "m"))
    ... except Exception: pass

    """
    name = f" {name}" if name else name
    msg = f"The input{name} must be greater than {comparison_name}."
    return eqx.error_if(x, jnp.any(x <= min_val), msg)


def check_greater_than_equal(
    x: AbstractQuantity,
    min_val: AbstractQuantity,
    /,
    *,
    name: str = "",
    comparison_name: str = "the specified minimum value",
) -> AbstractQuantity:
    """Check that the input value is greater than or equal to the input minimum value.

    Examples
    --------
    >>> import unxt as u

    Pass through the input if the value is greater than or equal to the min value:

    >>> x = u.Quantity([1, 2, 3], "m")
    >>> check_greater_than_equal(x, u.Quantity(1, "m"))
    Quantity['length'](Array([1, 2, 3], dtype=int32), unit='m')

    Raise an error if the input is smaller than the minimum value.

    >>> try: check_greater_than_equal(x, u.Quantity(2, "m"))
    ... except Exception: pass

    """
    name = f" {name}" if name else name
    msg = f"The input{name} must be greater than or equal to {comparison_name}."
    return eqx.error_if(x, jnp.any(x < min_val), msg)
