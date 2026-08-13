r"""The hyperbolic angle between two timelike directions.

`~coordinax.manifolds.angle_between` refuses a timelike pair, and correctly:
$g(u,u)$ and $g(v,v)$ are both negative, so ``arccos`` of their ratio would clip
to $0$ or $\pi$ and report two observers in relative motion as parallel.  The
invariant that *does* separate them is hyperbolic -- the relative rapidity -- so
it gets its own name rather than an extra branch inside a circular-angle
function.

Gated on `~coordinax.manifolds.AbstractLorentzianMetricField` like the causal
verbs: a metric with no timelike direction has no timelike vectors to measure
between.
"""

__all__: tuple[str, ...] = ()

from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import plum

import quaxed.numpy as qnp
import unxt as u
from unxt.quantity import AllowValue

import coordinaxs.api.manifolds as cxmapi
from coordinax._src.base import (
    AbstractChart,
    AbstractLorentzianMetricField,
    AbstractMetricField,
)
from coordinax._src.custom_types import OptUSys
from coordinaxs.api.custom_types import CDict

if TYPE_CHECKING:
    from jaxtyping import Array
from coordinax._src.manifolds.quadratic_form import gram

#: Slack on ``cosh(phi) >= 1``, so ordinary float error at coincident
#: directions (where the exact value is 1) is not read as a sign error.
_COSH_ATOL = 1e-6

_MSG_NOT_TIMELIKE = (
    "rapidity_between is defined only between two timelike tangent vectors: "
    "g(u,u) and g(v,v) must both be negative, and here they are {uu} and {vv}. "
    "For a spacelike pair the invariant is a circular angle -- use "
    "`angle_between`."
)

_MSG_OPPOSED = (
    "rapidity_between is undefined for two oppositely time-oriented vectors: "
    "one points to the future and the other to the past, so -g(u,v) is "
    "negative and there is no boost carrying one to the other. Negate one of "
    "them to compare their time-reverses."
)

_MSG_NOT_LORENTZIAN = (
    "rapidity_between() requires a Lorentzian metric -- *exactly one* timelike "
    "direction, signature (-1, 1, ..., 1) -- because a rapidity is the "
    "hyperbolic angle between two timelike vectors, measured against the single "
    "time orientation they share. {name} has signature {sig}. Use "
    "`angle_between` for the Riemannian angle."
)


def _check_rapidity_is_defined(
    u_timelike: Any, v_timelike: Any, cosh: Any, uu: Any, vv: Any, /
) -> None:
    """Raise unless a real rapidity exists between the two vectors.

    Takes the predicates already computed by the caller rather than recomputing
    them, so the eager check and the traced mask cannot drift apart. ``uu`` and
    ``vv`` are here only to be quoted, with units, in the message.

    Skipped under JAX tracing, where the values are not concrete; the caller
    applies the same conditions as a mask there, yielding `nan` instead.
    """
    if any(isinstance(x, jax.core.Tracer) for x in (u_timelike, v_timelike, cosh)):  # ty: ignore[possibly-missing-submodule]
        return

    if not (bool(jnp.all(u_timelike)) and bool(jnp.all(v_timelike))):
        raise ValueError(_MSG_NOT_TIMELIKE.format(uu=uu, vv=vv))
    # Both timelike, so the reverse Cauchy-Schwarz inequality gives
    # |cosh| >= 1 -- the only way to miss the branch is the wrong sign.
    if bool(jnp.any(cosh < 1.0 - _COSH_ATOL)):
        raise ValueError(_MSG_OPPOSED)


@plum.dispatch
def rapidity_between(
    metric: AbstractLorentzianMetricField,
    chart: AbstractChart,
    uvec: CDict,
    vvec: CDict,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> Any:
    r"""Relative rapidity between two timelike tangent vectors.

    .. math::

        \cosh\phi = \frac{-g(u,v)}{\sqrt{g(u,u)\,g(v,v)}}

    Dimensionless, like `~coordinax.transforms.LorentzBoost.rapidity`: every
    unit cancels in the ratio. For two four-velocities it is the rapidity of the
    boost carrying one frame to the other, so $\tanh\phi$ is their relative
    speed in units of $c$ and $\cosh\phi$ is the Lorentz factor $\gamma$.

    Rapidity is the parameterisation that *adds* under composition, which is
    what makes it the natural invariant here -- see the velocity-addition
    section of the special-relativity tutorial.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> at = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}
    >>> def four_velocity(beta):
    ...     g = 1.0 / jnp.sqrt(1.0 - beta**2)
    ...     return {"ct": u.Q(g, ""), "x": u.Q(g * beta, ""),
    ...             "y": u.Q(0.0, ""), "z": u.Q(0.0, "")}

    An observer at rest has zero rapidity relative to itself:

    >>> rest = four_velocity(0.0)
    >>> cxm.lorentzian.rapidity_between(cxc.minkowskict, rest, rest, at=at).round(6)
    Array(0., dtype=float64)

    Against a frame moving at $0.6c$ it is $\mathrm{arctanh}\,0.6$:

    >>> phi = cxm.lorentzian.rapidity_between(
    ...     cxc.minkowskict, rest, four_velocity(0.6), at=at)
    >>> bool(jnp.allclose(phi, jnp.arctanh(0.6), atol=1e-6))
    True

    and $\tanh$ takes it back to the relative speed, while $\cosh$ gives
    $\gamma$:

    >>> round(float(jnp.tanh(phi)), 4), round(float(jnp.cosh(phi)), 4)
    (0.6, 1.25)

    Rapidities add where velocities do not -- two $0.6c$ frames are separated by
    $2\,\mathrm{arctanh}\,0.6$, not by $\mathrm{arctanh}\,1.2$:

    >>> back = four_velocity(-0.6)
    >>> phi2 = cxm.lorentzian.rapidity_between(
    ...     cxc.minkowskict, back, four_velocity(0.6), at=at)
    >>> bool(jnp.allclose(phi2, 2 * jnp.arctanh(0.6), atol=1e-6))
    True

    A spacelike vector has no rapidity; that pair wants an ordinary angle:

    >>> xhat = {"ct": u.Q(0.0, ""), "x": u.Q(1.0, ""),
    ...         "y": u.Q(0.0, ""), "z": u.Q(0.0, "")}
    >>> try:
    ...     cxm.lorentzian.rapidity_between(cxc.minkowskict, rest, xhat, at=at)
    ... except ValueError as e:
    ...     print(str(e).split(":")[0])
    rapidity_between is defined only between two timelike tangent vectors

    """
    del metric  # the contraction reads the chart's metric, validated by `gram`
    chart.check_data(at, keys=True, values=False)
    chart.check_data(uvec, keys=True, values=False)
    chart.check_data(vvec, keys=True, values=False)

    # One metric evaluation for all three contractions, exactly as
    # `angle_between` does. `require_usys=False`: cosh(phi) is a ratio, so the
    # units cancel and the result is dimensionless whatever the inputs carry.
    inner, uu, vv = gram(
        uvec, vvec, chart, at=at, usys=usys, fname="rapidity_between",
        require_usys=False,
    )  # fmt: skip

    # `ustrip(AllowValue, "")` converts, so it *refuses* anything not actually
    # dimensionless, and lets a bare-array caller through untouched. The ratio's
    # units cancel exactly; the sign tests compare against zero *in* whatever
    # unit g(v,v) carries, so only their booleans are converted.
    cosh = cast("Array", u.ustrip(AllowValue, "", -inner / qnp.sqrt(uu * vv)))
    u_timelike = cast("Array", u.ustrip(AllowValue, "", uu < 0))
    v_timelike = cast("Array", u.ustrip(AllowValue, "", vv < 0))

    # Eagerly this raises, naming the case. Under tracing it cannot -- the
    # values are not concrete -- so `valid` below is what stands between the
    # caller and a wrong answer.
    _check_rapidity_is_defined(u_timelike, v_timelike, cosh, uu, vv)

    valid = u_timelike & v_timelike & (cosh >= 1.0 - _COSH_ATOL)
    # The clamp is float-error insurance for the *valid* branch only, where the
    # exact value is >= 1; `valid` has already excluded the genuine sign errors.
    return jnp.where(valid, jnp.arccosh(jnp.maximum(cosh, 1.0)), jnp.nan)


@plum.dispatch
def rapidity_between(
    chart: AbstractChart,
    uvec: CDict,
    vvec: CDict,
    /,
    *,
    at: CDict,
    usys: OptUSys = None,
) -> Any:
    """Relative rapidity, resolving the metric from the chart.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> at = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}
    >>> rest = {"ct": u.Q(1.0, ""), "x": u.Q(0.0, ""),
    ...         "y": u.Q(0.0, ""), "z": u.Q(0.0, "")}
    >>> cxm.lorentzian.rapidity_between(cxc.minkowskict, rest, rest, at=at).round(6)
    Array(0., dtype=float64)

    """
    return cxmapi.rapidity_between(chart.M.metric, chart, uvec, vvec, at=at, usys=usys)


@plum.dispatch
def rapidity_between(
    metric: AbstractMetricField,
    chart: AbstractChart,
    uvec: CDict,
    vvec: CDict,
    /,
    **kw: Any,
) -> Any:
    """Refuse a non-Lorentzian metric, by name.

    Examples
    --------
    >>> import unxt as u
    >>> import coordinax.charts as cxc
    >>> import coordinax.manifolds as cxm

    >>> at = {k: u.Q(0.0, "m") for k in ("x", "y", "z")}
    >>> v = {"x": u.Q(1.0, ""), "y": u.Q(0.0, ""), "z": u.Q(0.0, "")}
    >>> try:
    ...     cxm.lorentzian.rapidity_between(cxc.cart3d, v, v, at=at)
    ... except NotImplementedError as e:
    ...     print(str(e).split("--")[0].strip())
    rapidity_between() requires a Lorentzian metric

    """
    del chart, uvec, vvec, kw
    raise NotImplementedError(
        _MSG_NOT_LORENTZIAN.format(
            name=type(metric).__name__, sig=tuple(metric.signature)
        )
    )
