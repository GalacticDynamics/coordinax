r"""`nearest_tau` must return a *minimum* of the distance, never a maximum.

The residual $\mathbf{T}\cdot(\mathbf{x}-\gamma)$ vanishes at every stationary
point.  Across a genuine minimum it crosses positive-to-negative; across a
maximum it crosses negative-to-positive.  A bracket that merely contains *a*
sign change can therefore hold a maximum, which bisection will happily find and
report as a success.

That happens whenever the curve has structure finer than the seed spacing --
here 32 wiggles across the bounds against `n_seed=64`.
"""

import jax.numpy as jnp
import numpy as np

import unxt as u

import coordinaxs.curveframes as cxfc

BOUNDS = (u.Q(0.0, "s"), u.Q(10.0, "s"))


def wiggly(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """A curve whose wiggle period is finer than the default seed spacing."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([t, 0.3 * jnp.sin(20 * t), jnp.zeros_like(t)]), "km")


def _dist(tau_v: float) -> float:
    """Distance from the probe point to the curve at ``tau_v``."""
    g = np.asarray(wiggly(u.Q(tau_v, "s")).ustrip("km"))
    return float(np.linalg.norm(PROBE - g))


PROBE = np.array([7.192838530482548, -0.21267041401162823, 0.0])


def test_it_does_not_return_a_local_maximum() -> None:
    """The reported point must be a minimum, not the maximum next door."""
    builder = cxfc.FrenetSerretBuilder(wiggly, "s")
    x = u.Q(jnp.asarray(PROBE), "km")

    try:
        tau = float(cxfc.nearest_tau(builder, x, bounds=BOUNDS).ustrip("s"))
    except RuntimeError:
        return  # refusing is a correct outcome; lying is not

    # Second derivative of dist^2 must be positive: a minimum, not a maximum.
    h = 1e-5
    d2 = (_dist(tau + h) ** 2 - 2 * _dist(tau) ** 2 + _dist(tau - h) ** 2) / h**2
    assert d2 > 0, f"tau={tau} is a stationary point with d2(dist^2)={d2} <= 0"


def test_it_is_never_worse_than_the_coarse_scan_it_started_from() -> None:
    """Whatever the solve does, it may not return a point worse than its seed.

    The scan's argmin is available for free, so returning something farther
    away than it is strictly a regression -- and is exactly what landing on a
    maximum looks like.
    """
    builder = cxfc.FrenetSerretBuilder(wiggly, "s")
    x = u.Q(jnp.asarray(PROBE), "km")

    seeds = np.linspace(*(float(b.ustrip("s")) for b in BOUNDS), 64)
    best_seed = min(_dist(s) for s in seeds)

    try:
        tau = float(cxfc.nearest_tau(builder, x, bounds=BOUNDS).ustrip("s"))
    except RuntimeError:
        return

    assert _dist(tau) <= best_seed + 1e-9, (
        f"returned tau={tau} at distance {_dist(tau):.6f}, "
        f"worse than the coarse scan's own {best_seed:.6f}"
    )
