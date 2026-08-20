"""Representation of coordinates in different systems."""

__all__ = ("polar_range", "strictly_positive", "leq", "geq")

from typing import Any

import equinox as eqx
import jax

import quaxed.numpy as jnp
import unxt as u
from unxt import AbstractQuantity as AbcQ

from coordinax._src.exceptions import MismatchedManifoldError

_0d = u.Angle(jnp.array(0), "rad")
_pid = u.Angle(jnp.array(180), "deg")


def polar_range(polar: AbcQ, _l: AbcQ = _0d, _u: AbcQ = _pid, /) -> AbcQ:
    """Check that the polar angle is in the range.

    >>> import unxt as u

    Pass through the input if it's in the range.

    >>> x = u.Q([0., 1, 2], "deg")
    >>> polar_range(x)
    Q([0., 1., 2.], 'deg')

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
        u.ustrip("", jnp.any(jnp.logical_or((polar < _l), (polar > _u)))),
        "The inclination angle must be in the range [0, pi].",
    )


def strictly_positive(
    x: u.AbstractQuantity, /, *, name: str = ""
) -> u.AbstractQuantity:
    """Check that the input is non-negative and non-zero.

    >>> import unxt as u

    Pass through the input if the value is non-negative.

    >>> x = u.Q([1, 2, 3], "m")
    >>> strictly_positive(x)
    Q([1, 2, 3], 'm')

    Raise an error if any value is negative or zero.

    >>> x = u.Q([-1, 1, 2], "m")
    >>> try: strictly_positive(x)
    ... except Exception as e: pass

    >>> x = u.Q([0, 1, 2], "m")
    >>> try: strictly_positive(x)
    ... except Exception as e: pass

    """
    name = f" {name}" if name else name
    pred = u.ustrip("", jnp.any(x <= 0))
    # TODO: enable error_if to work on non-tracers.
    if isinstance(pred, jax.core.Tracer):  # ty: ignore[possibly-missing-submodule]
        return eqx.error_if(
            x, pred, f"The input{name} must be non-negative and non-zero."
        )
    if bool(pred):  # concrete 0-d array -> Python bool
        msg = f"The input{name} must be non-negative and non-zero."
        raise ValueError(msg)
    return x


def leq(
    x: u.AbstractQuantity,
    max_val: u.AbstractQuantity,
    /,
    *,
    name: str = "",
    comp_name: str = "the specified maximum value",
) -> u.AbstractQuantity:
    """Check that the input value is less than or equal to the input maximum value.

    >>> import unxt as u

    Pass through the input if the value is less than or equal to the max value:

    >>> x = u.Q([1, 2, 3], "m")
    >>> leq(x, u.Q(3, "m"))
    Q([1, 2, 3], 'm')

    Raise an error if the input is larger than the maximum value.

    >>> try: leq(x, u.Q(2, "m"))
    ... except Exception: pass

    """
    name = f" {name}" if name else name
    msg = f"The input{name} must be less than or equal to {comp_name}."
    return eqx.error_if(x, u.ustrip("", jnp.any(x > max_val)), msg)


def geq(
    x: u.AbstractQuantity,
    min_val: u.AbstractQuantity,
    /,
    *,
    name: str = "",
    comp_name: str = "the specified minimum value",
) -> u.AbstractQuantity:
    """Check that the input value is greater than or equal to the input minimum value.

    >>> import unxt as u

    Pass through the input if the value is greater than or equal to the min value:

    >>> x = u.Q([1, 2, 3], "m")
    >>> geq(x, u.Q(1, "m"))
    Q([1, 2, 3], 'm')

    Raise an error if the input is smaller than the minimum value.

    >>> try: geq(x, u.Q(2, "m"))
    ... except Exception: pass

    """
    name = f" {name}" if name else name
    msg = f"The input{name} must be greater than or equal to {comp_name}."
    return eqx.error_if(x, u.ustrip("", jnp.any(x < min_val)), msg)


def check_manifolds_match_charts(
    from_M: Any, from_chart: Any, to_M: Any, to_chart: Any, /
) -> None:
    """Refuse a `pt_map` whose manifolds are not its charts' own.

    The rules take the manifolds explicitly as well as the charts. Naming one
    that is not the chart's own -- or asking across two manifolds, which no
    rule implements -- is refused here rather than by a bare ``assert``, which
    disappears under ``python -O``.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinax._src.charts.checks import check_manifolds_match_charts

    >>> check_manifolds_match_charts(cxm.R3, cxc.cart3d, cxm.R3, cxc.sph3d)

    >>> try:
    ...     check_manifolds_match_charts(cxm.R3, cxc.cart3d, cxm.R2, cxc.sph3d)
    ... except cxc.MismatchedManifoldError as e:
    ...     print(e)
    to_M Rn(2) is not to_chart's manifold Rn(3)

    """
    check_manifold_matches_chart(from_M, from_chart, "from_M")
    check_manifold_matches_chart(to_M, to_chart, "to_M")


def check_manifold_matches_chart(M: Any, chart: Any, label: str, /) -> None:
    """Refuse one manifold argument that is not its chart's own.

    The single-sided form, for the rules that take both manifolds but only use
    one -- `coordinaxs.curveframes` has two, each `del`-ing the other.

    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm
    >>> from coordinax._src.charts.checks import check_manifold_matches_chart

    >>> check_manifold_matches_chart(cxm.R3, cxc.sph3d, "to_M")

    >>> try:
    ...     check_manifold_matches_chart(cxm.R2, cxc.sph3d, "to_M")
    ... except cxc.MismatchedManifoldError as e:
    ...     print(e)
    to_M Rn(2) is not to_chart's manifold Rn(3)

    """
    if M != chart.M:
        msg = f"{label} {M} is not {label.replace('_M', '_chart')}'s manifold {chart.M}"
        raise MismatchedManifoldError(msg)
