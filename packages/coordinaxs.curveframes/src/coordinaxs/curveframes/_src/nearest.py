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
    one seed spacing either side of it*: the argmin is guaranteed within one
    spacing of the true minimiser, and the residual
    $\mathbf{T}\cdot(\mathbf{x}-\boldsymbol{\gamma})$ equals
    $-\|\gamma'\|^{-1}\,d/d\tau(\tfrac12\mathrm{dist}^2)$, which crosses from
    positive to negative across a genuine minimum, so that bracket is
    well-posed for bisection and cannot land on the maximum next door.

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
    unit = builder.tau_unit
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
    tau0 = seeds[jnp.argmin(jax.vmap(dist2)(seeds))]
    spacing = (hi - lo) / (n_seed - 1)
    bracket_lo, bracket_hi = tau0 - spacing, tau0 + spacing

    def residual(tau_v: jax.Array, args: Any) -> jax.Array:
        del args
        T = builder.rotation_matrix(u.Q(tau_v, unit))[0]
        return jnp.dot(T, offset(tau_v))

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
    bracket_has_root = (jnp.sign(r_lo) != jnp.sign(r_hi)) & (
        jnp.abs(r_hi - r_lo) > atol
    )
    value = jnp.where(bracket_has_root, bsol.value, nsol.value)
    not_converged = jnp.where(
        bracket_has_root,
        bsol.result != optx.RESULTS.successful,
        nsol.result != optx.RESULTS.successful,
    )

    # Must surface non-convergence, not return silently (hybrid form,
    # matching ``_src/charts/checks.py``). The return value MUST be threaded
    # through -- an unused `eqx.error_if` result is dead-code-eliminated and
    # the guard vanishes under `jit`.
    msg = "nearest-point solve did not converge"
    if isinstance(not_converged, jax.core.Tracer):
        value = eqx.error_if(value, not_converged, msg)
    elif bool(not_converged):
        raise RuntimeError(msg)
    return u.Q(value, unit)
