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

import jax
import jax.numpy as jnp
import optimistix as optx

import unxt as u

_SOLVER_CACHE: dict[tuple[float, float], optx.Newton] = {}


def _solver(rtol: float, atol: float) -> optx.Newton:
    # Newton instances are cheap but hashable-by-identity; reusing one keeps
    # `jit` from seeing a fresh static argument on every call.
    key = (rtol, atol)
    if key not in _SOLVER_CACHE:
        _SOLVER_CACHE[key] = optx.Newton(rtol=rtol, atol=atol)
    return _SOLVER_CACHE[key]


def nearest_tau(
    builder: Any,
    x: u.AbstractQuantity,
    /,
    *,
    bounds: tuple[u.AbstractQuantity, u.AbstractQuantity],
    n_seed: int = 64,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> u.AbstractQuantity:
    r"""Curve parameter of the point on the curve nearest to ``x``.

    A Newton solve alone finds *a* stationary point of the distance, not the
    nearest one: on a curve that doubles back, the answer depends entirely on
    the initial guess. So scan ``n_seed`` points across ``bounds`` first, take
    the global argmin, and polish that. The argmin is an integer index and
    carries no gradient; the polish is implicitly differentiated.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import unxt as u
    >>> import coordinaxs.curveframes as cxfc
    >>> from coordinaxs.curveframes._src.nearest import nearest_tau

    >>> def circle(tau):
    ...     t = tau.ustrip("s")
    ...     return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")

    >>> b = cxfc.BishopBuilder(circle)
    >>> tau = nearest_tau(b, u.Q(jnp.array([2.0, 0.0, 0.0]), "km"),
    ...                   bounds=(u.Q(-1.0, "s"), u.Q(6.0, "s")))
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

    sol = optx.root_find(residual, _solver(rtol, atol), tau0, max_steps=64, throw=False)
    return u.Q(sol.value, unit)
