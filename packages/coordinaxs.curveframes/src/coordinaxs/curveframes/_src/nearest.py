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


def _default_tol() -> float:
    """Solver tolerance scaled to the active dtype's working precision.

    `1e-10` is unreachable in float32 -- JAX's default dtype everywhere
    outside this repo's pytest config, which sets `JAX_ENABLE_X64=1`. Below
    the dtype's own epsilon, a solve can never satisfy the tolerance and
    reports `max_steps_reached` on every call even though its answer is
    already correct to that dtype's precision. Scaling by the dtype's
    epsilon keeps float64 tight while giving float32 a tolerance it can
    actually reach.
    """
    return float(jnp.finfo(jnp.zeros(()).dtype).eps) ** 0.5


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

    A Newton solve alone finds *a* stationary point of the distance, not the
    nearest one: on a curve that doubles back, the answer depends entirely on
    the initial guess. So scan ``n_seed`` points across ``bounds`` first, take
    the global argmin, and polish that. The argmin is an integer index and
    carries no gradient; the polish is implicitly differentiated.

    ``rtol``/``atol`` default to `None`, meaning "derive from the active
    dtype" (see `_default_tol`); pass either explicitly to override.

    Raises
    ------
    Exception
        If the Newton polish does not converge (`eqx.error_if` under `jit`,
        a plain exception when eager). This cannot detect the periodic-
        aliasing case for a closed curve queried outside a one-period
        `tau_bounds` -- that solve *converges*, just to the wrong branch; see
        `TubularChart.tau_bounds`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinaxs.curveframes as cxfc

    >>> def circle(tau):
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")

    >>> b = cxfc.BishopBuilder(circle)
    >>> tau = cxfc.nearest_tau(b, u.Q(jnp.array([2.0, 0.0, 0.0]), "km"),
    ...                        bounds=(u.Q(-1.0, "s"), u.Q(6.0, "s")))
    >>> bool(jnp.allclose(tau.ustrip("s"), 0.0, atol=1e-6))
    True

    """
    unit = builder.tau_unit
    lo = jnp.asarray(bounds[0].ustrip(unit), dtype=float)
    hi = jnp.asarray(bounds[1].ustrip(unit), dtype=float)
    x_unit = x.unit
    xv = jnp.asarray(x.ustrip(x_unit), dtype=float)

    def offset(tau_v: jax.Array) -> jax.Array:
        g = builder.location(u.Q(tau_v, unit))
        return xv - jnp.asarray(g.ustrip(x_unit), dtype=float)

    def dist2(tau_v: jax.Array) -> jax.Array:
        d = offset(tau_v)
        return jnp.sum(d * d)

    # 1. Coarse global scan -- this is what makes the answer the *nearest*.
    seeds = jnp.linspace(lo, hi, n_seed)
    tau0 = seeds[jnp.argmin(jax.vmap(dist2)(seeds))]

    # 2. Newton polish on the stationarity condition.
    def residual(tau_v: jax.Array, args: Any) -> jax.Array:
        del args
        T = builder.rotation_matrix(u.Q(tau_v, unit))[0]
        return jnp.dot(T, offset(tau_v))

    tol = _default_tol()
    # Built per call rather than cached: the solver is a trace-time constant,
    # so `jit` sees it once however it is produced -- measured, a fresh Newton
    # on every call still traces exactly once. Two Newtons with equal
    # tolerances also compare and hash equal, so caching buys nothing.
    solver = optx.Newton(
        rtol=tol if rtol is None else rtol, atol=tol if atol is None else atol
    )
    sol = optx.root_find(residual, solver, tau0, max_steps=64, throw=False)

    # Surface a non-converged solve rather than silently returning a wrong
    # tau. Hybrid form, matching `_src/charts/checks.py`: `eqx.error_if`
    # under trace (a Python `bool` on a tracer raises
    # `TracerBoolConversionError`), a plain exception when concrete. The
    # return value MUST be threaded back into what is returned -- a bare
    # `eqx.error_if(pred, pred, msg)` is dead-code-eliminated and the guard
    # silently disappears under `jit`.
    pred = sol.result != optx.RESULTS.successful
    msg = "nearest-point solve did not converge"
    if isinstance(pred, jax.core.Tracer):
        value = eqx.error_if(sol.value, pred, msg)
    elif bool(pred):
        raise RuntimeError(msg)
    else:
        value = sol.value
    return u.Q(value, unit)
