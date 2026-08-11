r"""Causal structure of a Lorentzian manifold.

These verbs mean nothing without a timelike direction: a pair of points has a
*causal character* -- timelike, null, or spacelike -- only when the metric has
signature $(-,+,\\ldots,+)$. They therefore dispatch on
`~coordinax.manifolds.AbstractLorentzianMetricField` rather than on a chart,
so a Riemannian metric has no method rather than a method that accepts and then
refuses.

They live in the Minkowski package for the same reason ``scale_factors`` and
``metric_matrix`` do: a verb is declared once in `coordinaxs.api`, and each
manifold registers its own dispatch. `~coordinax.manifolds.interval` stays in
the generic manifolds package, because the signed quadratic form *is* defined
for every metric -- it is the squared `~coordinax.manifolds.separation` when
Riemannian, and it is what these three read.
"""

__all__: tuple[str, ...] = ()

from typing import Any, Literal, cast

import plum

import quaxed.numpy as jnp
import unxt as u

import coordinaxs.api.manifolds as cxmapi
from coordinax._src.base import (
    AbstractChart,
    AbstractLorentzianMetricField,
    AbstractMetricField,
)
from coordinax._src.custom_types import CDict, OptUSys

#: Speed of light in vacuum, for converting a length-valued proper interval
#: into a duration.  `MinkowskiCT` measures time as ``ct`` in length units, so
#: this is needed only at the boundary where a caller wants seconds.
C_LIGHT = u.Q(299792458.0, "m/s")

CausalCharacter = Literal["timelike", "null", "spacelike"]

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


def _as_float(x: Any, /) -> float:
    """Return *x* as a plain float, stripping units only if it has any.

    `chart.check_data(..., values=False)` permits bare arrays alongside
    quantities, so a components-have-units assumption would break the
    bare-array path.
    """
    unit = u.unit_of(x)
    return float(x if unit is None else u.ustrip(unit, x))


def _classify(ds2: Any, a: CDict, b: CDict, keys: tuple[str, ...], atol: Any, /) -> str:
    """Classify a precomputed interval, so callers evaluate it only once."""
    ds2_val = _as_float(ds2)

    if atol is None:
        # Scale-free default: compare against the largest squared coordinate
        # difference, so "close to zero" means small *relative to the data*
        # rather than small in whatever unit the caller happened to use.
        scale = max((_as_float(b[k] - a[k]) ** 2 for k in keys), default=1.0)
        tol = 1e-8 * max(scale, 1.0)
    else:
        tol = _as_float(atol)

    if ds2_val < -tol:
        return "timelike"
    if ds2_val > tol:
        return "spacelike"
    return "null"


# ===================================================================
# causal_character


@plum.dispatch
def causal_character(
    metric: AbstractLorentzianMetricField,
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
    ds2 = cxmapi.interval(chart, a, b, usys=usys)
    return _classify(ds2, a, b, chart.components, atol)


# ===================================================================
# proper_time / proper_distance


@plum.dispatch
def proper_time(
    metric: AbstractLorentzianMetricField,
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
    # One evaluation of the interval, classified and then used. Calling
    # `causal_character` here would compute it a second time, which on a curved
    # metric means a second `metric_matrix` build.
    ds2 = cast("Any", cxmapi.interval(chart, a, b, usys=usys))
    kind = _classify(ds2, a, b, chart.components, atol)
    if kind != "timelike":
        raise ValueError(_MSG_NOT_TIMELIKE.format(kind=kind, ds2=ds2))
    return jnp.sqrt(-ds2) / C_LIGHT


@plum.dispatch
def proper_distance(
    metric: AbstractLorentzianMetricField,
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
    # One evaluation of the interval, classified and then used. Calling
    # `causal_character` here would compute it a second time, which on a curved
    # metric means a second `metric_matrix` build.
    ds2 = cast("Any", cxmapi.interval(chart, a, b, usys=usys))
    kind = _classify(ds2, a, b, chart.components, atol)
    if kind != "spacelike":
        raise ValueError(_MSG_NOT_SPACELIKE.format(kind=kind, ds2=ds2))
    return jnp.sqrt(ds2)


# ---------------------------------------------------------------------------
# Chart-level convenience: resolve the metric from the chart, then redispatch.
# Mirrors how `norm` and `separation` layer chart-level over metric-level. A
# chart whose manifold carries a non-Lorentzian metric finds no method on the
# redispatch, which is the point: the refusal is the type system's, not a
# runtime signature scan's.


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
    """Classify a pair of events, resolving the metric from the chart.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> o = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}
    >>> ev = {"ct": u.Q(5.0, "m"), "x": u.Q(1.0, "m"),
    ...       "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> cxm.causal_character(cxc.minkowskict, o, ev)
    'timelike'

    """
    return cast(
        "str",
        cxmapi.causal_character(chart.M.metric, chart, a, b, atol=atol, usys=usys),
    )


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
    """Proper time between two events, resolving the metric from the chart.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> o = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}
    >>> rest = {"ct": u.Q(299792458.0, "m"), "x": u.Q(0.0, "m"),
    ...         "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> cxm.proper_time(cxc.minkowskict, o, rest).uconvert("s").round(3)
    Q(1., 's')

    """
    return cxmapi.proper_time(chart.M.metric, chart, a, b, atol=atol, usys=usys)


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
    """Proper distance between two events, resolving the metric from the chart.

    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> o = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}
    >>> far = {"ct": u.Q(3.0, "m"), "x": u.Q(5.0, "m"),
    ...        "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> cxm.proper_distance(cxc.minkowskict, o, far).round(2)
    Q(4., 'm')

    """
    return cxmapi.proper_distance(chart.M.metric, chart, a, b, atol=atol, usys=usys)


# ---------------------------------------------------------------------------
# Readable refusal for a non-Lorentzian metric.
#
# Without these, a Riemannian metric hits plum's `NotFoundLookupError`, which is
# accurate -- there genuinely is no method -- but reports it as a resolution
# dump with the call echoed back. The type still does the work; these only
# translate "no method" into a sentence naming the requirement.

_MSG_NOT_LORENTZIAN = (
    "{fname}() requires a Lorentzian metric -- one timelike direction, "
    "signature (-1, 1, ..., 1) -- because it reads the *sign* of the interval. "
    "{name} has signature {sig}, under which every separation has the same "
    "character and there is nothing to classify. Use `interval` for the signed "
    "square, or `separation` for a Riemannian distance."
)


def _not_lorentzian(metric: AbstractMetricField, fname: str, /) -> str:
    return _MSG_NOT_LORENTZIAN.format(
        fname=fname, name=type(metric).__name__, sig=tuple(metric.signature)
    )


@plum.dispatch
def causal_character(
    metric: AbstractMetricField, chart: AbstractChart, a: CDict, b: CDict, /, **kw: Any
) -> str:
    """Refuse a non-Lorentzian metric, by name.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> a = {k: u.Q(0.0, "m") for k in ("x", "y", "z")}
    >>> b = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> try:
    ...     cxm.causal_character(cxc.cart3d, a, b)
    ... except NotImplementedError as e:
    ...     print(str(e).split("--")[0].strip())
    causal_character() requires a Lorentzian metric

    """
    del chart, a, b, kw
    raise NotImplementedError(_not_lorentzian(metric, "causal_character"))


@plum.dispatch
def proper_time(
    metric: AbstractMetricField, chart: AbstractChart, a: CDict, b: CDict, /, **kw: Any
) -> Any:
    """Refuse a non-Lorentzian metric, by name."""
    del chart, a, b, kw
    raise NotImplementedError(_not_lorentzian(metric, "proper_time"))


@plum.dispatch
def proper_distance(
    metric: AbstractMetricField, chart: AbstractChart, a: CDict, b: CDict, /, **kw: Any
) -> Any:
    """Refuse a non-Lorentzian metric, by name."""
    del chart, a, b, kw
    raise NotImplementedError(_not_lorentzian(metric, "proper_distance"))
