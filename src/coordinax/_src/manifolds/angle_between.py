"""Dispatch implementations for :func:`coordinaxs.api.manifolds.angle_between`."""

__all__: tuple[str, ...] = ()


import jax
import jax.numpy as jnp
import plum

import quaxed.numpy as qnp
import unxt as u

import coordinax.angles as cxa
import coordinaxs.api.charts as cxcapi
import coordinaxs.api.manifolds as cxmapi
from ._utils import as_quantity_matrix
from coordinax._src.base import AbstractChart, AbstractMetricField
from coordinax._src.custom_types import CDict, OptUSys
from coordinax._src.metric.matrix import DenseMetric


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
    chart.check_data(at, keys=True, values=False)
    chart.check_data(uvec, keys=True, values=False)
    chart.check_data(vvec, keys=True, values=False)

    mm = cxmapi.metric_matrix(chart.M, at, chart)
    g = as_quantity_matrix(
        mm.matrix if isinstance(mm, DenseMetric) else mm.to_dense().matrix  # ty: ignore[unresolved-attribute]
    )
    u_qm = cxcapi.carray(uvec, chart.components)
    v_qm = cxcapi.carray(vvec, chart.components)

    inner = u_qm @ (g @ v_qm)
    uu = u_qm @ (g @ u_qm)
    vv = v_qm @ (g @ v_qm)

    _check_angle_is_defined(uu, vv, inner)

    cosine = inner / qnp.sqrt(uu * vv)
    # The clip is float-error insurance only. Under a positive-definite metric
    # Cauchy-Schwarz guarantees |cosine| <= 1; under an indefinite one it does
    # not, which is why `_check_angle_is_defined` verifies it rather than
    # letting the clip quietly flatten an out-of-range value to 0 or pi.
    cosine_value = qnp.clip(u.ustrip("", cosine), -1.0, 1.0)
    return cxa.Angle(qnp.arccos(cosine_value), "rad")


_MSG_NULL = (
    "angle_between is undefined for null (zero-norm) tangent vectors: the "
    "cosine's denominator vanishes."
)

_MSG_TIMELIKE = (
    "angle_between is undefined for two timelike tangent vectors: g(u,u) and "
    "g(v,v) are both negative, and the invariant separating them is a "
    "*hyperbolic* angle -- the relative rapidity, arccosh(-g(u,v)/sqrt(g(u,u) "
    "g(v,v))) -- not a circular angle. Computing `arccos` here would clip to 0 "
    "or pi and silently report no relative motion."
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


def _check_angle_is_defined(
    uu: u.AbstractQuantity, vv: u.AbstractQuantity, inner: u.AbstractQuantity, /
) -> None:
    r"""Raise unless a real circular angle exists between the two vectors.

    A metric angle needs the metric to be positive-definite *on the plane the
    two vectors span* -- which is a condition on the arguments, not on the
    metric. Under a Riemannian metric it always holds, so none of these fire.
    Under a Lorentzian one it holds exactly for a spacelike pair spanning a
    spacelike plane, which is the case this function serves.

    Skipped under JAX tracing, where the values are not concrete.
    """
    values = [q.value for q in (uu, vv, inner)]
    if any(isinstance(x, jax.core.Tracer) for x in values):  # ty: ignore[possibly-missing-submodule]
        return

    uu_v = float(jnp.min(jnp.asarray(uu.value)))
    vv_v = float(jnp.min(jnp.asarray(vv.value)))

    if bool(jnp.any(jnp.asarray(uu.value) == 0)) or bool(
        jnp.any(jnp.asarray(vv.value) == 0)
    ):
        raise ValueError(_MSG_NULL)
    if uu_v < 0 and vv_v < 0:
        raise ValueError(_MSG_TIMELIKE)
    if uu_v < 0 or vv_v < 0:
        raise ValueError(_MSG_MIXED)

    # Both spacelike, but their span may still be Lorentzian -- e.g. u=(0,1,0,0)
    # and v=(1,2,0,0) are each spacelike while their span contains (1,0,0,0).
    cos = jnp.asarray(u.ustrip("", inner / qnp.sqrt(uu * vv)))
    if bool(jnp.any(jnp.abs(cos) > 1.0 + 1e-6)):
        raise ValueError(_MSG_NOT_SPACELIKE_PLANE)
