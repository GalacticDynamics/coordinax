r"""Dispatch implementations for the interval / causality API.

`norm` refuses indefinite metrics, because $\sqrt{v^\top G v}$ has no real value
when the quadratic form can go negative.  The quantity that *is* well defined
there is the form itself, unrooted:

$$ \Delta s^2 = \Delta x^\top G\, \Delta x, $$

which is what this module exposes as `interval`.  For a Riemannian metric it is
simply the squared `separation`; for a Lorentzian one its **sign** is the causal
character of the pair, and its magnitude gives proper time (timelike) or proper
distance (spacelike).
"""

__all__: tuple[str, ...] = ()

from typing import Any, Literal, cast

import plum

import quaxed.numpy as jnp
import unxt as u

import coordinaxs.api.charts as cxcapi
import coordinaxs.api.manifolds as cxmapi
from .quadratic_form import quadratic_form
from coordinax._src.base import AbstractChart, AbstractMetricField
from coordinax._src.custom_types import CDict, OptUSys

#: Speed of light in vacuum, for converting a length-valued proper interval
#: into a duration.  `MinkowskiCT` measures time as ``ct`` in length units, so
#: this is needed only at the boundary where a caller wants seconds.
C_LIGHT = u.Q(299792458.0, "m/s")

CausalCharacter = Literal["timelike", "null", "spacelike"]

_MSG_NOT_LORENTZIAN = (
    "{fname}() requires a Lorentzian metric (exactly one negative signature "
    "entry, as in Minkowski's (-1, 1, 1, 1)); {name} has signature {sig}. "
    "Causal character is not defined for this metric."
)

_MSG_NOT_TIMELIKE = (
    "proper_time() is defined only for timelike-separated events; this pair is "
    "{kind} (interval^2 = {ds2}). Use `interval` for the signed square, or "
    "`proper_distance` for a spacelike pair."
)

_MSG_NOT_SPACELIKE = (
    "proper_distance() is defined only for spacelike-separated events; this "
    "pair is {kind} (interval^2 = {ds2}). Use `interval` for the signed "
    "square, or `proper_time` for a timelike pair."
)


def _require_lorentzian(metric: AbstractMetricField, fname: str, /) -> None:
    """Raise unless *metric* has exactly one timelike direction."""
    sig = tuple(metric.signature)
    if sum(1 for s in sig if s < 0) != 1:
        msg = _MSG_NOT_LORENTZIAN.format(
            fname=fname, name=type(metric).__name__, sig=sig
        )
        raise ValueError(msg)


# ===================================================================
# interval


@plum.dispatch
def interval(
    chart: AbstractChart, a: CDict, b: CDict, /, *, usys: OptUSys = None
) -> Any:
    r"""Signed squared interval between two points, in the chart's metric.

    Unlike `separation`, this is defined for *every* metric, because it never
    takes a square root.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    For a Riemannian metric it is the squared separation:

    >>> a = {"x": u.Q(3.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> b = {"x": u.Q(0.0, "m"), "y": u.Q(4.0, "m"), "z": u.Q(0.0, "m")}
    >>> cxm.interval(cxc.cart3d, a, b).round(2)
    Q(25., 'm2')

    For Minkowski it is negative for a timelike pair -- the case that used to
    make `separation` return ``nan``:

    >>> o = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}
    >>> ev = {"ct": u.Q(5.0, "m"), "x": u.Q(1.0, "m"),
    ...       "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> cxm.interval(cxc.minkowskict, o, ev).round(2)
    Q(-24., 'm2')

    """
    return cxmapi.interval(chart.M.metric, chart, a, b, usys=usys)


@plum.dispatch
def interval(
    metric: AbstractMetricField,
    chart: AbstractChart,
    a: CDict,
    b: CDict,
    /,
    *,
    usys: OptUSys = None,
) -> Any:
    r"""Signed squared interval with respect to an explicit metric.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> o = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}
    >>> ev = {"ct": u.Q(1.0, "m"), "x": u.Q(5.0, "m"),
    ...       "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> cxm.interval(cxm.MinkowskiMetric(), cxc.minkowskict, o, ev).round(2)
    Q(24., 'm2')

    """
    chart.check_data(a, keys=True, values=False)
    chart.check_data(b, keys=True, values=False)

    diff = {k: b[k] - a[k] for k in chart.components}

    # The same contraction `norm` takes the square root of, evaluated at `a`.
    # Sharing it is what gives `interval` the unit handling it would otherwise
    # have to restate -- and makes `separation**2 == interval` hold by
    # construction rather than by the test that asserts it.
    return quadratic_form(diff, chart, at=a, usys=usys, fname="interval")


@plum.dispatch
def interval(
    chart: AbstractChart,
    a: u.AbstractQuantity,
    b: u.AbstractQuantity,
    /,
    *,
    usys: OptUSys = None,
) -> Any:
    """Signed squared interval for packed `unxt.Quantity` vectors.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> a = u.Q([0.0, 0.0, 0.0, 0.0], "m")
    >>> b = u.Q([5.0, 1.0, 0.0, 0.0], "m")
    >>> cxm.interval(cxc.minkowskict, a, b).round(2)
    Q(-24., 'm2')

    """
    return cxmapi.interval(
        chart, cxcapi.cdict(a, chart), cxcapi.cdict(b, chart), usys=usys
    )


# ===================================================================
# causal_character


@plum.dispatch
def causal_character(
    chart: AbstractChart,
    a: CDict,
    b: CDict,
    /,
    *,
    atol: Any = None,
    usys: OptUSys = None,
) -> str:
    r"""Classify a pair of events as ``"timelike"``, ``"null"``, or ``"spacelike"``.

    The classification is the sign of `interval`, with ``atol`` deciding how
    close to zero counts as null (default: ``1e-8`` of the largest squared
    coordinate difference, so it scales with the data).

    This returns a Python `str` and so is **not** ``jit``-able; it is a
    diagnostic. Inside a traced computation, branch on the sign of `interval`.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> o = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}
    >>> def ev(ct, x):
    ...     return {"ct": u.Q(ct, "m"), "x": u.Q(x, "m"),
    ...             "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}

    A big time difference and a small space difference is timelike -- the two
    events can be visited by one slower-than-light observer:

    >>> cxm.causal_character(cxc.minkowskict, o, ev(5.0, 1.0))
    'timelike'

    The reverse is spacelike -- no observer sees both:

    >>> cxm.causal_character(cxc.minkowskict, o, ev(1.0, 5.0))
    'spacelike'

    Equal parts is null: exactly a light ray.

    >>> cxm.causal_character(cxc.minkowskict, o, ev(3.0, 3.0))
    'null'

    """
    _require_lorentzian(chart.M.metric, "causal_character")
    ds2 = cxmapi.interval(chart, a, b, usys=usys)

    unit = u.unit_of(ds2)
    ds2_val = float(u.ustrip(unit, ds2)) if unit is not None else float(ds2)

    if atol is None:
        # Scale-free default: compare against the largest squared coordinate
        # difference, so "close to zero" means small *relative to the data*
        # rather than small in whatever unit the caller happened to use.
        diffs = (
            float(u.ustrip(u.unit_of(b[k]), b[k] - a[k])) ** 2 for k in chart.components
        )
        scale = max(diffs, default=1.0)
        tol = 1e-8 * max(scale, 1.0)
    else:
        has_unit = u.unit_of(atol) is not None
        tol = float(u.ustrip(unit, atol)) if has_unit else float(atol)

    if ds2_val < -tol:
        return "timelike"
    if ds2_val > tol:
        return "spacelike"
    return "null"


# ===================================================================
# proper_time / proper_distance


@plum.dispatch
def proper_time(
    chart: AbstractChart,
    a: CDict,
    b: CDict,
    /,
    *,
    atol: Any = None,
    usys: OptUSys = None,
) -> Any:
    r"""Proper time elapsed between two timelike-separated events.

    For a timelike pair, $c\,\tau = \sqrt{-\Delta s^2}$; this returns the
    duration $\tau$, dividing by $c$ so the result is a *time*, not the
    length-valued $c\tau$ that the ``ct`` chart works in.

    Raises `ValueError` for a null or spacelike pair, where no observer's
    wristwatch connects the two events.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    An observer at rest for 1 light-second of coordinate time ages 1 second:

    >>> o = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}
    >>> rest = {"ct": u.Q(299792458.0, "m"), "x": u.Q(0.0, "m"),
    ...         "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> cxm.proper_time(cxc.minkowskict, o, rest).uconvert("s").round(3)
    Q(1., 's')

    A spacelike pair has no proper time:

    >>> far = {"ct": u.Q(1.0, "m"), "x": u.Q(5.0, "m"),
    ...        "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> try:
    ...     cxm.proper_time(cxc.minkowskict, o, far)
    ... except ValueError as e:
    ...     print(str(e)[:56])
    proper_time() is defined only for timelike-separated eve

    """
    kind = cxmapi.causal_character(chart, a, b, atol=atol, usys=usys)
    ds2 = cast("Any", cxmapi.interval(chart, a, b, usys=usys))
    if kind != "timelike":
        raise ValueError(_MSG_NOT_TIMELIKE.format(kind=kind, ds2=ds2))
    return jnp.sqrt(-ds2) / C_LIGHT


@plum.dispatch
def proper_distance(
    chart: AbstractChart,
    a: CDict,
    b: CDict,
    /,
    *,
    atol: Any = None,
    usys: OptUSys = None,
) -> Any:
    r"""Proper distance between two spacelike-separated events.

    For a spacelike pair, $\Delta\sigma = \sqrt{\Delta s^2}$ -- the distance
    measured in the frame where the two events are simultaneous.

    Raises `ValueError` for a null or timelike pair.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> o = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}
    >>> far = {"ct": u.Q(3.0, "m"), "x": u.Q(5.0, "m"),
    ...        "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> cxm.proper_distance(cxc.minkowskict, o, far).round(2)
    Q(4., 'm')

    """
    kind = cxmapi.causal_character(chart, a, b, atol=atol, usys=usys)
    ds2 = cast("Any", cxmapi.interval(chart, a, b, usys=usys))
    if kind != "spacelike":
        raise ValueError(_MSG_NOT_SPACELIKE.format(kind=kind, ds2=ds2))
    return jnp.sqrt(ds2)
