"""Representation of coordinates in different systems."""

__all__ = ("polar_range", "strictly_positive", "leq", "geq")

import equinox as eqx

import quaxed.numpy as jnp
import unxt as u
from unxt import AbstractQuantity as AbcQ

_0d = u.Angle(jnp.array(0), "rad")
_pid = u.Angle(jnp.array(180), "deg")


def polar_range(polar: AbcQ, _l: AbcQ = _0d, _u: AbcQ = _pid, /) -> AbcQ:
    """Check that the polar angle is in the range.

    Examples
    --------
    >>> import unxt as u
    >>> from coordinax.charts._src.checks import polar_range

    Pass through the input if it's in the range.

    >>> x = u.Q([0., 1, 2], "deg")
    >>> polar_range(x)
    Quantity(Array([0., 1., 2.], dtype=float64), unit='deg')

    Raise an error if anything is outside the range.

    >>> x = u.Q([0., 1, 2], "m")
    >>> try: polar_range(x)
    ... except Exception as e: print("wrong units")
    wrong units

    >>> x = u.Q([-1., 1, 2], "deg")
    >>> try: polar_range(x)
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


def strictly_positive(
    x: u.AbstractQuantity, /, *, name: str = ""
) -> u.AbstractQuantity:
    """Check that the input is non-negative and non-zero.

    Examples
    --------
    >>> import unxt as u

    Pass through the input if the value is non-negative.

    >>> x = u.Q([1, 2, 3], "m")
    >>> strictly_positive(x)
    Quantity(Array([1, 2, 3], dtype=int64), unit='m')

    Raise an error if any value is negative or zero.

    >>> x = u.Q([-1, 1, 2], "m")
    >>> try: strictly_positive(x)
    ... except Exception as e: pass

    >>> x = u.Q([0, 1, 2], "m")
    >>> try: strictly_positive(x)
    ... except Exception as e: pass

    """
    name = f" {name}" if name else name
    return eqx.error_if(
        x, jnp.any(x <= 0), f"The input{name} must be non-negative and non-zero."
    )


def leq(
    x: u.AbstractQuantity,
    max_val: u.AbstractQuantity,
    /,
    *,
    name: str = "",
    comp_name: str = "the specified maximum value",
) -> u.AbstractQuantity:
    """Check that the input value is less than or equal to the input maximum value.

    Examples
    --------
    >>> import unxt as u

    Pass through the input if the value is less than or equal to the max value:

    >>> x = u.Q([1, 2, 3], "m")
    >>> leq(x, u.Q(3, "m"))
    Quantity(Array([1, 2, 3], dtype=int64), unit='m')

    Raise an error if the input is larger than the maximum value.

    >>> try: leq(x, u.Q(2, "m"))
    ... except Exception: pass

    """
    name = f" {name}" if name else name
    msg = f"The input{name} must be less than or equal to {comp_name}."
    return eqx.error_if(x, jnp.any(x > max_val), msg)


def geq(
    x: u.AbstractQuantity,
    min_val: u.AbstractQuantity,
    /,
    *,
    name: str = "",
    comp_name: str = "the specified minimum value",
) -> u.AbstractQuantity:
    """Check that the input value is greater than or equal to the input minimum value.

    Examples
    --------
    >>> import unxt as u

    Pass through the input if the value is greater than or equal to the min value:

    >>> x = u.Q([1, 2, 3], "m")
    >>> geq(x, u.Q(1, "m"))
    Quantity(Array([1, 2, 3], dtype=int64), unit='m')

    Raise an error if the input is smaller than the minimum value.

    >>> try: geq(x, u.Q(2, "m"))
    ... except Exception: pass

    """
    name = f" {name}" if name else name
    msg = f"The input{name} must be greater than or equal to {comp_name}."
    return eqx.error_if(x, jnp.any(x < min_val), msg)
