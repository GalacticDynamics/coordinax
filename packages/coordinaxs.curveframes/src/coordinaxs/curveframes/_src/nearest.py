r"""Nearest-point projection onto a parameterised curve.

Given an ambient point $\mathbf{x}$, the curve parameter of the closest point
satisfies the stationarity condition

$$ \mathbf{T}(\tau)\cdot(\mathbf{x} - \boldsymbol{\gamma}(\tau)) = 0, $$

i.e. the offset is orthogonal to the tangent. That is a scalar root-find, and
`optimistix` differentiates through it implicitly -- which is what keeps a
fitted curve's parameters differentiable through the chart transition.
"""

__all__ = ("nearest_tau",)


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx

import unxt as u


def nearest_tau(
    builder: Any,
    x: u.AbstractQuantity,
    /,
    *,
    bounds: tuple[u.AbstractQuantity, u.AbstractQuantity],
    n_seed: int = 64,
    rtol: float | None = None,
    atol: float | None = None,
) -> u.AbstractQuantity:
    r"""Curve parameter of the point on the curve nearest to ``x``.

    An unconstrained root-find alone finds *a* stationary point of the
    distance, not the nearest one -- and not even always a minimum: the
    stationarity condition below is satisfied at every local maximum of the
    distance too, and a solve left free to wander can walk out of the basin
    the scan found and converge onto one of those instead (measured: a
    sine-wave curve with ``x=(4.4, -1.4, 0)`` km sends an unconstrained
    Newton polish 3.2 seed spacings from a correctly-chosen start onto a
    maximum 2.4x farther away, reporting success). So scan ``n_seed`` points
    across ``bounds`` first, take the global argmin, and root-find *within
    one seed spacing either side of it*: the argmin is within one spacing of
    the true minimiser **provided ``n_seed`` resolves the curve** -- it is an
    assumption on the scan, not a guarantee, and a curve that wiggles faster
    than the spacing breaks it. The residual
    $\mathbf{T}\cdot(\mathbf{x}-\boldsymbol{\gamma})$ equals
    $-\|\gamma'\|^{-1}\,d/d\tau(\tfrac12\mathrm{dist}^2)$, which crosses from
    positive to negative across a genuine minimum, so that bracket is
    well-posed for bisection and cannot land on the maximum next door.

    ``bounds`` must have non-zero width. A zero-width one makes the seed
    spacing zero, and the bracketed solve's ``expand_if_necessary`` grows a
    bracket by doubling its width, which never grows a zero -- so it is
    refused rather than left to loop forever -- with a `ValueError` eagerly, and
    as a `RuntimeError` (equinox's `error_if`) under `jit` or `vmap`, where the
    check cannot be a Python branch. That is a precondition, not a degradation:
    there is no curve to search.

    That bracket does not always contain a sign change, though -- the
    residual can be one-signed across it in two situations: the true nearest
    point lies outside `tau_bounds` altogether (the scan is confined to
    `tau_bounds`, so its argmin sits at the edge with the real root further
    out), or the query is genuinely degenerate, equidistant from the whole
    curve (e.g. the centre of a circular curve), where there is no
    particular nearest point at all. An unconstrained Newton polish from the
    scan's argmin is used for that case instead, exactly as before this
    bracket existed -- it walks to the correct answer in the first situation
    and, because its own derivative vanishes identically on a degenerate
    query, fails to converge in the second, which is what tells the two
    apart. The mainline case above never reaches this fallback, since its
    bracket is guaranteed to contain a sign change. The scan's argmin is an
    integer index and carries no gradient; both root-finds are implicitly
    differentiated.

    ``rtol``/``atol`` default to `None`, meaning "derive from the active
    dtype's epsilon" (below); pass either explicitly to override.

    Raises
    ------
    Exception
        If neither solve converges (`eqx.error_if` under `jit`, a plain
        exception when eager). This cannot detect the periodic-aliasing case
        for a closed curve queried outside a one-period `tau_bounds` -- that
        solve *converges*, just to the wrong branch; see
        `TubularChart.tau_bounds`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinaxs.curveframes as cxfc

    >>> def circle(tau):
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")

    >>> b = cxfc.BishopBuilder(circle, "s")
    >>> tau = cxfc.nearest_tau(b, u.Q(jnp.array([2.0, 0.0, 0.0]), "km"),
    ...                        bounds=(u.Q(0.0, "s"), u.Q(2 * jnp.pi, "s")))
    >>> bool(jnp.allclose(tau.ustrip("s"), 0.0, atol=1e-6))
    True

    """
    # The bounds are the tau range, so they carry tau's unit themselves -- no
    # need to consult the builder, which may not have one declared. Reading
    # them is not merely a shortcut: `_tau_unit_at` prefers a *declared*
    # `tau_unit`, and on a pinned-station builder that describes the station
    # while these bounds are times, so consulting it scans seconds in
    # kilometres. The builder is the fallback for bare (unitless) bounds only.
    if n_seed < 2:
        msg_seed = (
            f"`n_seed` must be at least 2, got {n_seed}: the scan needs a spacing "
            "to bracket around, and one point has none. Below that the failure is "
            "a divide-by-zero in the spacing and an empty grid, which surfaces as "
            "an unrelated shape error."
        )
        raise ValueError(msg_seed)

    unit = u.unit_of(bounds[0])
    if unit is None:
        unit = builder._tau_unit_at(bounds[0])
    # `jnp.asarray` narrows only here: `ustrip` is typed as a broad union, and
    # `ty` rejects `hi - lo` between two of them. Everywhere else the bare
    # `ustrip` is enough.
    lo = jnp.asarray(bounds[0].ustrip(unit))
    hi = jnp.asarray(bounds[1].ustrip(unit))
    x_unit = x.unit
    xv = x.ustrip(x_unit)

    def offset(tau_v: jax.Array) -> jax.Array:
        g = builder.location(u.Q(tau_v, unit))
        return xv - g.ustrip(x_unit)

    def dist2(tau_v: jax.Array) -> jax.Array:
        d = offset(tau_v)
        return jnp.sum(d * d)

    # 1. Coarse global scan -- this is what makes the answer the *nearest*.
    seeds = jnp.linspace(lo, hi, n_seed)
    scan = jax.vmap(dist2)(seeds)
    i_best = jnp.argmin(scan)
    tau0, d_seed = seeds[i_best], scan[i_best]
    spacing = (hi - lo) / (n_seed - 1)

    # A zero-width `bounds` makes `spacing` zero, and `Bisection`'s
    # `expand_if_necessary` below grows a bracket by *doubling its width* --
    # doubling zero never grows it, and that expansion is not bounded by
    # `max_steps`, so the solve loops with no exit instead of failing
    # (measured: still running at 45 s, where a proper interval returns in
    # seconds). Guard here, where every caller routes through: `TubularChart`
    # validates the *dimensions* of `tau_bounds` but not that they differ, so a
    # degenerate chart reaches this on every inverse `pt_map`. Threaded through
    # `spacing` so the check cannot be eliminated ahead of the bracket that
    # depends on it.
    #
    # Two exception types, because the check cannot be a Python branch on a
    # tracer: `ValueError` eagerly, `RuntimeError` (equinox's `error_if`) under
    # `jit` or `vmap`. The docstring says so, and the tests pin `RuntimeError`,
    # which both satisfy.
    degenerate = spacing == 0
    msg_bounds = (
        "`bounds` has zero width, so there is no curve to search: the "
        "nearest-point scan needs `bounds[0] != bounds[1]`."
    )
    if isinstance(degenerate, jax.core.Tracer):
        spacing = eqx.error_if(spacing, degenerate, msg_bounds)
    elif bool(degenerate):
        raise ValueError(msg_bounds)

    def residual(tau_v: jax.Array, args: Any) -> jax.Array:
        del args
        # `tangent()`, not `rotation_matrix()[0]`: both builders override it to
        # skip the parallel-transport solve only rows 1-2 need, and document the
        # value as identical. Measured bit-identical and ~110x faster eagerly,
        # which is what makes the fine residual grid below affordable.
        T = builder.tangent(u.Q(tau_v, unit)).ustrip("")
        return jnp.dot(T, offset(tau_v))

    # 1b. Narrow the bracket before solving. `+/- spacing` is two spacings wide,
    # so once the curve varies on that scale it can hold a whole period -- two
    # minima and two maxima. Bisection then returns *a* root with the right
    # endpoint orientation, which may be the worse one: measured on a curve
    # with 32 wiggles across `bounds`, a bracket of width 0.31746 against a
    # period of 0.31416 held minima at 8.34701 (distance 0.107) and 8.46367
    # (distance 0.011), and the solve returned the first.
    #
    # Refined on the **residual**, not on `dist2`. Those coincide only when the
    # tangent is the unit tangent of the parametrisation; on a station-pinned
    # worldtube it is the curve's *spatial* tangent while `tau` is a time, so
    # `dist2`'s minimum sits away from the root. Narrowing around the `dist2`
    # argmin there excluded the true root and turned a passing round trip into
    # a refusal. Every crossing with the minimum's orientation is a candidate;
    # the closest one wins, which is what makes the multi-minimum bracket
    # resolve to the *nearest* rather than to whichever bisection reaches.
    fine = jnp.linspace(tau0 - spacing, tau0 + spacing, n_seed)

    # `residual` and `dist2` each call `offset`, so this looks like it evaluates
    # the curve twice per grid point. It does not once compiled: XLA eliminates
    # the duplicate as a common subexpression. Hand-fusing them into one
    # tuple-returning `vmap` measured *slower* in both regimes -- eager 0.619s
    # against 0.373s, jit compile 0.99s against 0.73s, warm call 0.046ms
    # against 0.042ms -- so the obvious optimisation is a pessimisation here.
    r_fine = jax.vmap(lambda t: residual(t, None))(fine)
    d_fine = jax.vmap(dist2)(fine)

    crossing = (r_fine[:-1] > 0) & (r_fine[1:] < 0)
    # Rank candidates by the distance across the crossing, not by residual.
    score = jnp.where(crossing, 0.5 * (d_fine[:-1] + d_fine[1:]), jnp.inf)
    k = jnp.argmin(score)
    found = jnp.any(crossing)
    # No crossing on the fine grid keeps the original coarse bracket, so the
    # documented degradations -- nearest point outside `bounds`, and a
    # genuinely degenerate query -- reach the unconstrained fallback as before.
    bracket_lo = jnp.where(found, fine[k], tau0 - spacing)
    bracket_hi = jnp.where(found, fine[k + 1], tau0 + spacing)
    # `d_seed` deliberately stays the *coarse* minimum. Tightening it with the
    # fine grid rejected correct answers: the post-check below assumes the
    # answer is no worse than the best sampled point, which holds only when
    # the objective is `dist2`. On a station-pinned worldtube the chart
    # inverse is the perpendicular foot, whose distance can exceed the
    # sampled minimum -- measured 0.000400 against a fine-grid minimum of
    # 0.000397, which refused a round trip that had always worked.

    # Scale by the dtype's epsilon, not a fixed `1e-10`: float32 (JAX's
    # default outside this repo's x64 pytest config) can never satisfy
    # `1e-10` below its own epsilon, and reports `max_steps_reached` on
    # every call despite an already-correct answer.
    tol = float(jnp.finfo(jnp.zeros(()).dtype).eps) ** 0.5
    rtol = tol if rtol is None else rtol
    atol = tol if atol is None else atol

    # 2a. Bracketed root-find within one seed spacing of the argmin (see the
    # docstring above). `expand_if_necessary=True` only silences
    # `Bisection.init`'s own error on a rootless bracket -- `bracket_has_root`
    # below re-checks that itself and routes to 2b, discarding `bsol` then.
    bisector = optx.Bisection(  # ty: ignore[missing-argument]
        rtol=rtol, atol=atol, flip="detect", expand_if_necessary=True
    )
    bsol = optx.root_find(
        residual,
        bisector,
        tau0,
        options={"lower": bracket_lo, "upper": bracket_hi},
        max_steps=64,
        throw=False,
    )

    # 2b. Unconstrained Newton, used only as a fallback -- see the docstring
    # above for the two cases (nearest point outside `tau_bounds`, or a
    # genuinely degenerate query) that leave the bracket above without a
    # sign change, and how this fallback's own convergence tells them apart.
    newton = optx.Newton(rtol=rtol, atol=atol)
    nsol = optx.root_find(residual, newton, tau0, max_steps=64, throw=False)

    # `atol`-gated, not sign-only: on a degenerate query (e.g. a circle's
    # centre) both endpoints are ~1e-16 noise that can land on either side of
    # zero, especially under `jit` where XLA's eval order differs from eager.
    r_lo, r_hi = residual(bracket_lo, None), residual(bracket_hi, None)
    # A sign change alone is not enough. The residual crosses positive-to-
    # negative across a minimum and negative-to-positive across a maximum, so
    # `jnp.sign(r_lo) != jnp.sign(r_hi)` -- which this used to test -- accepts
    # the maximum next door, and bisection then finds it and reports success.
    # Measured on a curve wiggling faster than the seed spacing: a returned
    # point 15.6x farther away than the true nearest, with the forward map
    # round-tripping to 3e-15 from the wrong labels so nothing downstream
    # noticed. Requiring the minimum's orientation is what the docstring above
    # has always claimed this test does.
    bracket_has_minimum = (r_lo > 0) & (r_hi < 0) & (jnp.abs(r_hi - r_lo) > atol)
    value = jnp.where(bracket_has_minimum, bsol.value, nsol.value)
    not_converged = jnp.where(
        bracket_has_minimum,
        bsol.result != optx.RESULTS.successful,
        nsol.result != optx.RESULTS.successful,
    )

    # The coarse argmin is already paid for, so a solve that lands farther away
    # than its own seed has failed whatever status it reports. This catches the
    # unconstrained fallback wandering onto a maximum -- and also the bracketed
    # branch when the scan is under-resolved: the endpoint orientation says only
    # that the residual falls across the bracket, not that the bracket holds a
    # single stationary point, so on a curve that wiggles *within* one spacing
    # bisection can still settle on an interior maximum. Either way the answer
    # is refused rather than returned.
    not_converged = not_converged | (dist2(value) > d_seed * (1.0 + rtol) + atol**2)

    # Must surface non-convergence, not return silently (hybrid form,
    # matching ``_src/charts/checks.py``). The return value MUST be threaded
    # through -- an unused `eqx.error_if` result is dead-code-eliminated and
    # the guard vanishes under `jit`.
    msg = (
        "nearest-point solve did not converge. The documented causes are: the "
        "query is degenerate, equidistant from the whole curve (e.g. a circular "
        "curve's centre), so no nearest point exists; the true nearest point "
        "lies outside `bounds`, which the scan cannot see past; or the curve "
        "varies faster than `n_seed` samples resolve, so the scan's argmin is "
        "not within one spacing of the true minimiser, and the bracket can hold "
        "more than one minimum. Only the last has a remedy here -- "
        "raise `n_seed` (measured: a curve with 32 wiggles across `bounds` "
        "refuses at the default 64 and resolves correctly at 128)."
    )
    if isinstance(not_converged, jax.core.Tracer):
        value = eqx.error_if(value, not_converged, msg)
    elif bool(not_converged):
        raise RuntimeError(msg)
    return u.Q(value, unit)
