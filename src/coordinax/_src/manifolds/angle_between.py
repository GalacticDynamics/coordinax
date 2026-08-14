"""Dispatch implementations for :func:`coordinaxs.api.manifolds.angle_between`."""

__all__: tuple[str, ...] = ()


from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import plum

import quaxed.numpy as qnp
import unxt as u
from unxt.quantity import AllowValue

import coordinax.angles as cxa
import coordinaxs.api.manifolds as cxmapi
from .quadratic_form import gram
from coordinax._src.base import (
    AbstractChart,
    AbstractMetricField,
    check_metric_is_charts,
)
from coordinax._src.custom_types import OptUSys
from coordinaxs.api.custom_types import CDict

if TYPE_CHECKING:
    from jaxtyping import Array


@plum.dispatch
def angle_between(
    chart: AbstractChart,
    uvec: CDict,
    vvec: CDict,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> cxa.AbstractAngle:
    """Manifold-level dispatch: delegate to the attached metric.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> at = {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m")}
    >>> uvec = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m")}
    >>> vvec = {"x": u.Q(0.0, "m"), "y": u.Q(1.0, "m")}
    >>> cxm.angle_between(cxc.cart2d, uvec, vvec, at=at)
    Angle(1.57079633, 'rad')

    """
    return cxmapi.angle_between(chart.M.metric, chart, uvec, vvec, at=at, usys=usys)  # ty: ignore[invalid-return-type]


@plum.dispatch
def angle_between(
    metric: AbstractMetricField,
    chart: AbstractChart,
    uvec: CDict,
    vvec: CDict,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> cxa.AbstractAngle:
    """Return the metric angle between two tangent vectors.

    The input component dictionaries are interpreted as tangent-vector
    components in the coordinate basis of ``chart``. The metric is evaluated at
    the base point ``at``.

    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> metric = cxm.FlatMetric(3)
    >>> at = {
    ...     "r": u.Q(2.0, "m"),
    ...     "theta": u.Angle(jnp.pi / 2, "rad"),
    ...     "phi": u.Angle(0.0, "rad"),
    ... }
    >>> uvec = {"r": u.Q(0.0, "m"), "theta": u.Angle(1.0, "rad"),
    ...         "phi": u.Angle(0.0, "rad")}
    >>> vvec = {"r": u.Q(0.0, "m"), "theta": u.Angle(0.0, "rad"),
    ...         "phi": u.Angle(1.0, "rad")}
    >>> cxm.angle_between(metric, cxc.sph3d, uvec, vvec, at=at)
    Angle(1.57079633, 'rad')

    An indefinite metric is not rejected outright. Two *spacelike* directions
    span a plane on which the Minkowski metric is positive-definite, so the
    angle between them is an ordinary one:

    >>> at4 = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}
    >>> xhat = {"ct": u.Q(0.0, "m"), "x": u.Q(1.0, "m"),
    ...         "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> yhat = {"ct": u.Q(0.0, "m"), "x": u.Q(0.0, "m"),
    ...         "y": u.Q(1.0, "m"), "z": u.Q(0.0, "m")}
    >>> cxm.angle_between(cxc.minkowskict, xhat, yhat, at=at4)
    Angle(1.57079633, 'rad')

    Two *timelike* directions have no circular angle between them -- the
    invariant is a hyperbolic one, the relative rapidity -- so this raises
    rather than clipping ``arccos`` to a meaningless value:

    >>> obs = {"ct": u.Q(1.0, "m"), "x": u.Q(0.0, "m"),
    ...        "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> moving = {"ct": u.Q(1.25, "m"), "x": u.Q(0.75, "m"),
    ...           "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
    >>> try:
    ...     cxm.angle_between(cxc.minkowskict, obs, moving, at=at4)
    ... except ValueError as e:
    ...     print(str(e).split(":")[0])
    angle_between is undefined for two timelike tangent vectors

    """
    check_metric_is_charts(metric, chart, "angle_between")
    chart.check_data(at, keys=True, values=False)
    chart.check_data(uvec, keys=True, values=False)
    chart.check_data(vvec, keys=True, values=False)

    # All three contractions from one metric evaluation, via the shared
    # primitive -- so the unit handling (mixed Quantity/bare-array rejection,
    # per-component units surviving the contraction) is the same one `norm`
    # uses, and the diagonal fast path comes along too. `gram` rather than three
    # `bilinear_form` calls: those would rebuild the metric matrix three times,
    # free under `jit` but ~1.6x eagerly.
    #
    # `require_usys=False`: the cosine is a ratio, so every unit cancels and the
    # result is dimensionless whatever the inputs carry. The "declare a unit
    # system" contract that `norm` and `interval` impose -- where the result's
    # unit *is* derived from the inputs -- would be ceremony here, and would
    # break callers that have always passed bare arrays.
    inner, uu, vv = gram(
        uvec, vvec, chart, at=at, usys=usys, fname="angle_between", require_usys=False
    )

    cos = u.ustrip(AllowValue, "", inner / qnp.sqrt(uu * vv))

    # Compared against zero *in* whatever unit g(v,v) carries -- the one
    # comparison needing no common unit -- so only the resulting boolean is
    # converted, and `AllowValue` lets a bare-array caller through untouched.
    u_spacelike = cast("Array", u.ustrip(AllowValue, "", uu > 0))
    v_spacelike = cast("Array", u.ustrip(AllowValue, "", vv > 0))

    # Eagerly this raises, naming the case. Under tracing it cannot -- the values
    # are not concrete -- so it is a no-op there and `valid` below is what stands
    # between the caller and a wrong answer.
    _check_angle_is_defined(uu, vv, cos)

    # The traced counterpart of the same conditions. Without it, `jit`/`vmap`
    # would reach the clip and hand back 0 or pi for a hyperbolic or imaginary
    # case -- silently reporting "anti-parallel" for two observers in relative
    # motion. `nan` is the honest value: no real angle exists.
    valid = u_spacelike & v_spacelike & (jnp.abs(cos) <= 1.0 + _COS_ATOL)
    # The clip is float-error insurance for the *valid* branch only; `valid`
    # has already excluded everything genuinely out of range.
    angle = jnp.arccos(jnp.clip(cos, -1.0, 1.0))
    return cxa.Angle(jnp.where(valid, angle, jnp.nan), "rad")


#: Slack on |cos| <= 1, so ordinary float error in a valid Riemannian case is
#: not mistaken for a genuinely non-spacelike plane.
_COS_ATOL = 1e-6

_MSG_NULL = (
    "angle_between is undefined for null (zero-norm) tangent vectors: the "
    "cosine's denominator vanishes."
)

_MSG_TIMELIKE = (
    "angle_between is undefined for two timelike tangent vectors: g(u,u) and "
    "g(v,v) are both negative, and the invariant separating them is a "
    "*hyperbolic* angle, not a circular one. Computing `arccos` here would clip "
    "to 0 or pi and silently report no relative motion. Use "
    "`coordinax.manifolds.lorentzian.rapidity_between`."
)

_MSG_MIXED = (
    "angle_between is undefined between a timelike and a spacelike tangent "
    "vector: g(u,u) g(v,v) < 0, so the cosine's denominator is imaginary."
)

_MSG_NOT_SPACELIKE_PLANE = (
    "angle_between is undefined here: both tangent vectors are spacelike, but "
    "they span a plane that is not, so |g(u,v)| > sqrt(g(u,u) g(v,v)) and the "
    "ratio is not a cosine. This cannot happen under a positive-definite "
    "metric, where Cauchy-Schwarz guarantees the bound."
)


def _check_angle_is_defined(uu: Any, vv: Any, cos: Any, /) -> None:
    r"""Raise unless a real circular angle exists between the two vectors.

    A metric angle needs the metric to be positive-definite *on the plane the
    two vectors span* -- which is a condition on the arguments, not on the
    metric. Under a Riemannian metric it always holds, so none of these fire.
    Under a Lorentzian one it holds exactly for a spacelike pair spanning a
    spacelike plane, which is the case this function serves.

    Skipped under JAX tracing, where the values are not concrete; the caller
    applies the same conditions as a mask there, yielding `nan` instead.
    """
    u_neg, v_neg = u.ustrip(AllowValue, "", uu < 0), u.ustrip(AllowValue, "", vv < 0)
    u_zero, v_zero = (
        u.ustrip(AllowValue, "", uu == 0),
        u.ustrip(AllowValue, "", vv == 0),
    )
    if any(isinstance(x, jax.core.Tracer) for x in (u_neg, v_neg, cos)):  # ty: ignore[possibly-missing-submodule]
        return

    if bool(jnp.any(u_zero)) or bool(jnp.any(v_zero)):
        raise ValueError(_MSG_NULL)
    if bool(jnp.any(u_neg)) and bool(jnp.any(v_neg)):
        raise ValueError(_MSG_TIMELIKE)
    if bool(jnp.any(u_neg)) or bool(jnp.any(v_neg)):
        raise ValueError(_MSG_MIXED)

    # Both spacelike, but their span may still be Lorentzian -- e.g. u=(0,1,0,0)
    # and v=(1,2,0,0) are each spacelike while their span contains (1,0,0,0).
    if bool(jnp.any(jnp.abs(jnp.asarray(cos)) > 1.0 + _COS_ATOL)):
        raise ValueError(_MSG_NOT_SPACELIKE_PLANE)
