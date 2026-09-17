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


def _check_query(x: u.AbstractQuantity, n_seed: int, /) -> None:
    """Refuse a query the scan cannot answer, naming the mistake.

    Both failures would otherwise surface from inside the scan as shape errors
    about `n_seed`-sized internals, nowhere near the call that caused them.
    Shapes are static, so this holds under `jit`, and under `vmap` it sees the
    per-element shape -- which is how a batch of query points is meant to be
    mapped.
    """
    if n_seed < 2:
        msg = (
            f"`n_seed` must be at least 2, got {n_seed}: the scan needs a spacing "
            "to bracket around, and one point has none. Below that the failure is "
            "a divide-by-zero in the spacing and an empty grid, which surfaces as "
            "an unrelated shape error."
        )
        raise ValueError(msg)

    if jnp.ndim(x) != 1:
        msg = (
            f"`x` must be a single ambient point, got shape {jnp.shape(x)}. "
            "Map a batch of query points with `jax.vmap(lambda p: nearest_tau("
            "builder, p, bounds=bounds))` rather than passing them together."
        )
        raise ValueError(msg)


def _relative_speed_floor(fallback: jax.Array) -> jax.Array:
    """`sqrt(eps)` of the fallback's *own* dtype, times the fallback.

    Not the default float: under ``jax_enable_x64`` that is f64 even for f32
    curve data, flooring 2.3e4x too low to clamp f32 noise.

    Only conditioning rides on this -- ``safe_speed`` stays positive either
    way, so the residual's sign and the root are unchanged. Hence the test is
    on this helper rather than on a solve.
    """
    return jnp.sqrt(jnp.finfo(fallback.dtype).eps) * fallback


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
    than the spacing breaks it. The residual handed to the solvers is
    $\mathbf{T}\cdot(\mathbf{x}-\boldsymbol{\gamma})$ over the speed
    $\|\gamma'\|$ -- floored below at a small multiple of the problem's own
    characteristic speed, so a station momentarily at rest cannot divide it by
    zero. Wherever that floor does not bind it equals
    $-\|\gamma'\|^{-2}\,d/d\tau(\tfrac12\mathrm{dist}^2)$ for a regular curve.
    It crosses from positive to negative across a genuine minimum, so that
    bracket is well-posed for bisection and cannot land on the maximum next
    door.

    The division by the speed leaves the root and its sign untouched -- the
    divisor is positive whether or not the floor binds -- and is there so the
    residual is measured in $\tau$ rather than in whatever length the ambient
    point carries, since ``atol`` is compared against both. Without it the same
    geometry converged differently in km and in m.

    ``bounds`` must have non-zero width. A zero-width one makes the seed
    spacing zero, and the bracketed solve's ``expand_if_necessary`` grows a
    bracket by doubling its width, which never grows a zero -- so it is
    refused rather than left to loop forever -- with a `ValueError` eagerly, and
    as a `RuntimeError` (equinox's `error_if`) under `jit` or `vmap`, where the
    check cannot be a Python branch. That is a precondition, not a degradation:
    there is no curve to search.

    ``bounds`` must also ascend: ``bounds[0] < bounds[1]``, refused the same
    two ways. A descending pair happens to answer correctly, but the bracket's
    minimum test and `ArcLength`'s `s_max` margin both read the pair as
    ascending, so nothing promises it will keep doing so.

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

    >>> b = cxfc.BishopBuilder(circle, "s", initial_normal="auto")
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
    # kilometres.
    #
    # No fallback for bare bounds: `unit_of` returns `None` only for a
    # non-quantity, and `bounds` is consumed by `.ustrip` just below, which a
    # non-quantity has not got -- so the branch could never have completed a
    # call. (`bounds`' declared type says the same, but only catches it when
    # `COORDINAX_ENABLE_RUNTIME_TYPECHECKING` is on, which it is not by
    # default.)
    _check_query(x, n_seed)

    unit = u.unit_of(bounds[0])
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
    tau0 = seeds[i_best]
    spacing = (hi - lo) / (n_seed - 1)

    # Zero `spacing` hangs the solve rather than failing it: `Bisection`'s
    # `expand_if_necessary` grows a bracket by doubling, which never grows
    # zero, and that loop is not bounded by `max_steps`. Guarded here, where
    # every caller routes through, and threaded through `spacing` so it cannot
    # be eliminated ahead of the bracket. Two exception types because the check
    # cannot be a Python branch on a tracer.
    #
    # Reversed bounds work today -- `spacing` goes negative and the grid runs
    # backwards -- but `bracket_has_minimum` and `_S_MAX_MARGIN` both assume
    # ascending. Accidentally correct is not a contract.
    msg_zero = (
        "`bounds` has zero width, so there is no curve to search: the "
        "nearest-point scan needs `bounds[0] != bounds[1]`."
    )
    msg_reversed = (
        "`bounds` runs backwards: `bounds[0]` must be below `bounds[1]`. "
        "Swap them. The scan, the bracket's minimum test and the arc-length "
        "margin all read the pair as ascending, so a descending one is "
        "outside what they promise even where it happens to answer correctly."
    )
    for bad, msg in ((spacing == 0, msg_zero), (spacing < 0, msg_reversed)):
        if isinstance(bad, jax.core.Tracer):
            spacing = eqx.error_if(spacing, bad, msg)
        elif bool(bad):
            raise ValueError(msg)

    # Hoisted out of a loop that runs ~130 times. `offset`, not `-offset`: the
    # norm below removes the sign.
    d_offset = jax.jacfwd(offset)

    # A characteristic length-per-tau for the problem, used only where the local
    # speed vanishes. The scan's spread of distances is a length and `hi - lo` a
    # tau, so the ratio has the units the residual needs. Floored so a query
    # sitting exactly on the curve cannot make it zero in turn.
    _extent = jnp.sqrt(jnp.max(scan)) - jnp.sqrt(jnp.min(scan))
    # `ones_like` keeps the fallback in the scan's own dtype instead of leaning
    # on JAX's weak-typing rules to do it.
    _fallback_speed = jnp.where(_extent > 0, _extent, jnp.ones_like(_extent)) / jnp.abs(
        hi - lo
    )
    # Floored relatively, not only at exactly zero. A station slowing to rest
    # passes through arbitrarily small speeds, and dividing by one of those
    # amplifies the residual without bound. Any positive scaling leaves the root
    # and its sign untouched, so a floor can only improve conditioning -- it
    # cannot move the answer.
    _speed_floor = _relative_speed_floor(_fallback_speed)

    def residual(tau_v: jax.Array, args: Any) -> jax.Array:
        del args
        # `tangent()`, not `rotation_matrix()[0]`: identical value, ~110x
        # faster eagerly, which is what makes the fine grid below affordable.
        T = builder.tangent(u.Q(tau_v, unit)).ustrip("")
        # Divided by |d(offset)/dtau| to put the residual in `tau` rather than
        # in the ambient length, so one `atol` is not a tolerance on both at
        # once. Scaling the tolerances instead does not work: the sensitivity
        # is inside the solvers' own convergence tests.
        #
        # `T` comes from `builder.tangent`, NOT from this derivative. On a
        # station-pinned worldtube they differ -- spatial tangent against
        # station velocity -- and reusing one for both redefines the root.
        speed = jnp.linalg.norm(d_offset(tau_v))
        # `speed` can be exactly zero while `T` is well defined: a station at
        # rest zeroes its velocity, not the slice's spatial tangent. The floor
        # is therefore a length-per-tau, not a bare `1.0`, or the division
        # above stops removing the ambiguity it exists for.
        safe_speed = jnp.maximum(speed, _speed_floor)
        return jnp.dot(T, offset(tau_v)) / safe_speed

    # 1b. Narrow the bracket first. `+/- spacing` is two spacings wide, so a
    # curve varying on that scale can hold a whole period, and bisection then
    # returns *a* root with the right orientation rather than the nearest one.
    #
    # Refined on the **residual**, not `dist2`: they coincide only when the
    # tangent is the parametrisation's, and on a station-pinned worldtube it is
    # the spatial tangent while `tau` is a time. Every crossing with the
    # minimum's orientation is a candidate and the closest wins.
    fine = jnp.linspace(tau0 - spacing, tau0 + spacing, n_seed)

    # Not a double evaluation once compiled: XLA eliminates the shared
    # `offset` as a common subexpression. Hand-fusing them measured slower.
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

    # From `tau0.dtype`, not the global default float: a tolerance below the
    # working dtype's resolution can never be met, and the solve reports
    # `max_steps_reached` on an already-correct answer (measured, 15 of 19 f32
    # queries under x64). Mixed precision is not covered -- f64 `bounds` over
    # an f32 curve still refuses, because the f64 `hi - lo` reaches the
    # residual and leaves it f64-typed carrying f32 information. Pass `bounds`
    # in the curve's own precision, or an explicit `atol`.
    tol = float(jnp.finfo(tau0.dtype).eps) ** 0.5
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

    # NOTE: #841 added a post-check here -- refuse when the answer is farther
    # away than the scan's own argmin. It is removed, because its premise is
    # false for a worldtube: the chart inverse there is the perpendicular foot,
    # not the nearest point, and the two differ once the station moves. It
    # refused correct round trips at n1 >= 0.08 km on the repo's own
    # `stretching` worldtube (dist2 0.006400 against a coarse seed of
    # 0.006373), and shipped only because the existing test used n1 = 0.02 km,
    # which is the last offset that passes.
    #
    # It also had no demonstrated true positive: on the wiggly curve it was
    # meant to catch, it passed on a 9.6x-wrong answer. The bracket refinement
    # above is what actually fixes that case.

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
