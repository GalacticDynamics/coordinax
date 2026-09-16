r"""How fast a slice's geometry deforms.

The extrinsic-curvature slot of the ADM decomposition,
$K_{ij} = \tfrac{1}{2\alpha}\left(\partial_t\gamma_{ij} - (\mathcal{L}_\beta
\gamma)_{ij}\right)$, reduces here to $\tfrac12\partial_t\gamma_{ij}$: the
lapse is identically 1 under absolute time, and the Lie-drag term vanishes
because a chart coordinate *is* what the time derivative holds fixed, so the
coordinate shift is zero by construction. See
`coordinaxs.curveframes.velocity` for the other ADM piece, which is the
ambient motion of the frame origin and does not vanish.
"""

__all__ = ("rate_of_strain",)

from collections.abc import Callable
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import unxts.linalg as ul

import unxt as u
from coordinaxs.api.manifolds import metric_matrix

_MSG_GAUGE = (
    "`rate_of_strain` is gauge-dependent off the curve axis (n1 or n2 is "
    "non-zero). Each slice's `(n1, n2)` labels are fixed by that slice's transport "
    "seed, which `BishopBuilder` picks from the *world* frame unless given an "
    "`initial_normal` -- so when the tangent rotates with time the labels name a "
    "different physical point on each slice, and differentiating reports the "
    "frame's drift as strain. Measured on a static helix, gamma_tau_tau spans "
    "1.040535 to 1.629908 across four seeds at n = (0.2, 0.1); on the axis it is "
    "1.16 for every seed. A rigid rotation, which is an isometry and must give "
    "K = 0, instead gives |K|max = 0.017405 about z-hat.\n\n"
    "On the axis (n = 0) the result is gauge-free and always valid. Off it, pass "
    "`assume_gauge_carried=True` only if your family carries one director through "
    "`initial_normal` rather than letting each slice choose. See #870."
)

_MSG_BATCHED_TIME = (
    "`rate_of_strain` differentiates at one time, so `t` must be a scalar; got "
    "shape {shape}. `K_ij` is a 2-tensor, and a batched `t` would make the "
    "Jacobian one rank higher -- `TubularChart` is single-point for the same "
    "reason. Use `jax.vmap` over scalar calls."
)

_MSG_BARE_TIME = (
    "`rate_of_strain` differentiates with respect to `t`, so `t` must carry a "
    "unit: nothing else states what the rate is per. Pass a `Quantity`, e.g. "
    "`u.Q(1.0, 's')`."
)


def rate_of_strain(
    chart_at_time: Callable[[Any], Any],
    point: dict,
    t: Any,
    /,
    *,
    assume_gauge_carried: bool = False,
) -> ul.QuantityMatrix:
    r"""Return $K_{ij} = \tfrac12\,\partial_t\gamma_{ij}$ at ``point``.

    Parameters
    ----------
    chart_at_time
        The slice family: given a time, the chart of the tube at that time.
        Typically ``lambda t: TubularChart(BishopBuilder(AtTime(curve, t), ...))``.
    point
        Chart coordinates at which to evaluate, e.g.
        ``{"tau": ..., "n1": ..., "n2": ...}``.
    t
        The time to differentiate at. Must carry a unit.
    assume_gauge_carried
        Opt out of the off-axis refusal. Set this only when the family carries
        one director across every slice -- by passing `initial_normal` rather
        than letting each `BishopBuilder` pick its own. It is an assertion by
        the caller, not something this can verify (#870).

    Notes
    -----
    **Longitudinal only.** The name invites the opposite expectation, but in
    exact arithmetic $K_{n_in_j}=0$ and $K_{\tau n_i}=0$, leaving
    $K_{\tau\tau}$ the only component that can carry anything. $n_1, n_2$ are
    Cartesian coordinates in an orthonormal normal plane, so
    $\gamma_{n_in_j}=\delta_{ij}$ at every offset whatever the tube does, and
    $\partial_t$ of a constant is zero.

    What comes back is that zero at floating-point noise, not exactly it: the
    metric is built from the Jacobian rather than read off symbolically, and
    $\partial_t$ runs through a `diffrax` solve. Measured on a curve that
    stretches and bends, against $K_{\tau\tau}=0.8176$: the normal block at
    ``2.8e-17``, the mixed block at ``7.5e-12``, and $\gamma_{n_in_j}$ within
    one ulp of $\delta_{ij}$. Compare against a tolerance, not against zero.

    Radial inflation and cross-sectional shear are not represented here at
    all; a tube doubling in radius reports the same $K$ as one holding it.
    What this measures is the rate the *labelling along the curve* stretches.

    The value depends on the *labelling*, and correctly so. A material
    parametrisation shows the tube genuinely stretching; wrapping the same
    curve in `ArcLength` holds the metric near unit-speed, so its rate of
    strain is near zero. That is the same choice `velocity` reports, made once
    by the caller's curve -- see the `ArcLength` docs.

    Takes the family rather than one chart because $\partial_t$ needs
    neighbouring slices, and a `TubularChart` is a single one.

    Off the curve axis the result depends on each slice's transport seed, so
    it is refused unless ``assume_gauge_carried`` says the caller has handled
    that. On the axis it is gauge-free and always valid. Measured: on a static
    helix ``gamma_tau_tau`` spans 1.040535 to 1.629908 across four seeds at
    ``n = (0.2, 0.1)`` and is 1.16 for every seed at ``n = 0``. The refusal is
    an `equinox.error_if`, so under `jax.jit` it fires when the compiled
    function runs rather than when it is traced; that is what keeps ``point``
    traceable at all.

    ``t`` must be a scalar. A batched one would raise the Jacobian's rank
    above 2, and `TubularChart` is single-point for the same reason; use
    `jax.vmap` over scalar calls. Left unguarded it failed inside the chart
    with ``All input arrays must have the same shape``, naming nothing.

    """
    t_unit = u.unit_of(t)
    if t_unit is None:
        raise TypeError(_MSG_BARE_TIME)
    shape = jnp.shape(t.value if t_unit is not None else t)
    if shape != ():
        raise ValueError(_MSG_BATCHED_TIME.format(shape=shape))

    def gamma(t_val: Any) -> Any:
        chart = chart_at_time(u.Q(t_val, t_unit))
        return cast("Any", metric_matrix(chart.M, point, chart)).matrix.value

    here = chart_at_time(t)
    unit = cast("Any", metric_matrix(here.M, point, here)).matrix.unit / t_unit

    d_gamma = jax.jacfwd(gamma)(t.ustrip(t_unit))

    if not assume_gauge_carried:
        # Deferred, and threaded through the value that is returned. A Python
        # `float(n)` here was the *only* thing stopping `jit`/`vmap` over
        # `point` -- the chart itself traces fine -- and an `error_if` whose
        # result goes unused is dead-code-eliminated under trace.
        off_axis = jnp.logical_or(
            u.ustrip(u.unit_of(point["n1"]), point["n1"]) != 0,
            u.ustrip(u.unit_of(point["n2"]), point["n2"]) != 0,
        )
        d_gamma = eqx.error_if(d_gamma, off_axis, _MSG_GAUGE)

    return ul.QuantityMatrix(0.5 * d_gamma, unit=unit)
