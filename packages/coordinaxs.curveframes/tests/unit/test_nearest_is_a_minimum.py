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
import pytest

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


def _nearest_or_refusal(builder: object, x: u.AbstractQuantity) -> float | None:
    """The reported tau, or `None` when `nearest_tau` itself declined to answer.

    A bare ``except RuntimeError`` would let an unrelated regression -- a
    failure inside the builder, say -- pass as an acceptable refusal, so match
    the stable opening of `nearest_tau`'s own non-convergence message.

    Matching on ``"n_seed"`` would not do: that message names all three
    documented causes and offers `n_seed` as the remedy for one of them, so the
    substring is present whichever cause fired. The prefix identifies the
    *source* of the refusal, which is what this needs to establish; which of the
    three causes fired is pinned separately by
    `test_the_remedy_the_refusal_advertises_actually_works`.
    """
    try:
        tau = cxfc.nearest_tau(builder, x, bounds=BOUNDS)
    except RuntimeError as exc:
        refusal = str(exc)
    else:
        return float(tau.ustrip("s"))
    assert refusal.startswith("nearest-point solve did not converge"), (
        f"refused, but not by `nearest_tau`'s documented path: {refusal}"
    )
    return None


def test_it_does_not_return_a_local_maximum() -> None:
    """The reported point must be a minimum, not the maximum next door."""
    builder = cxfc.FrenetSerretBuilder(wiggly, "s")
    x = u.Q(jnp.asarray(PROBE), "km")

    tau = _nearest_or_refusal(builder, x)
    if tau is None:
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

    tau = _nearest_or_refusal(builder, x)
    if tau is None:
        return

    assert _dist(tau) <= best_seed + 1e-9, (
        f"returned tau={tau} at distance {_dist(tau):.6f}, "
        f"worse than the coarse scan's own {best_seed:.6f}"
    )


def _distances(taus: np.ndarray) -> np.ndarray:
    """Distance from the probe to the curve, vectorised over ``taus``."""
    g = np.stack([taus, 0.3 * np.sin(20 * taus), np.zeros_like(taus)])
    return np.linalg.norm(PROBE[:, None] - g, axis=0)


def test_the_remedy_the_refusal_advertises_actually_works() -> None:
    """A scan that *does* resolve the curve must succeed, and be right.

    The refusal tells the caller to raise `n_seed`. Both tests above accept a
    refusal as a valid outcome, so on their own they would still pass if
    `nearest_tau` began refusing every query -- including ones where the scan
    resolves the curve perfectly well. This pins the advertised remedy.
    """
    builder = cxfc.FrenetSerretBuilder(wiggly, "s")
    x = u.Q(jnp.asarray(PROBE), "km")

    tau = float(cxfc.nearest_tau(builder, x, bounds=BOUNDS, n_seed=128).ustrip("s"))

    lo, hi = (float(b.ustrip("s")) for b in BOUNDS)
    grid = np.linspace(lo, hi, 200_001)
    truth = float(grid[int(np.argmin(_distances(grid)))])

    assert tau == pytest.approx(truth, abs=1e-3)
    # And it is genuinely the global minimum, not merely near a stationary point.
    assert _dist(tau) <= float(_distances(grid).min()) + 1e-6


# The probe from #847: the coarse bracket is 2 x spacing = 0.31746 wide against
# a wiggle period of 0.31416, so it held two minima -- 8.34701 at distance 0.107
# and 8.46367 at distance 0.011 -- and bisection returned the first. Both guards
# above pass on that answer: it is a genuine minimum, and it is closer than any
# seed. Only the bracket's *width* was wrong.
WRONG_MINIMUM_PROBE = np.array([8.45268, -0.10722, 0.0])


def test_it_picks_the_best_minimum_in_the_bracket_not_merely_one() -> None:
    """A bracket holding two minima must not yield the worse one."""
    builder = cxfc.FrenetSerretBuilder(wiggly, "s")
    x = u.Q(jnp.asarray(WRONG_MINIMUM_PROBE), "km")

    tau = float(cxfc.nearest_tau(builder, x, bounds=BOUNDS).ustrip("s"))

    lo, hi = (float(b.ustrip("s")) for b in BOUNDS)
    grid = np.linspace(lo, hi, 200_001)
    g = np.stack([grid, 0.3 * np.sin(20 * grid), np.zeros_like(grid)])
    best = float(np.min(np.linalg.norm(WRONG_MINIMUM_PROBE[:, None] - g, axis=0)))

    here = float(
        np.linalg.norm(
            WRONG_MINIMUM_PROBE - np.array([tau, 0.3 * np.sin(20 * tau), 0.0])
        )
    )
    # The other minimum in that bracket is 9.6x farther away; the solver's own
    # tolerance is what sets the slack here, not the choice of minimum.
    assert here <= best * 1.05, f"tau={tau} at {here:.6f}, best is {best:.6f}"


def test_an_ordinary_curve_is_not_refused_by_the_resolution_check() -> None:
    """The under-resolution guard must not fire on a well-resolved curve."""

    def circle(tau: u.AbstractQuantity) -> u.AbstractQuantity:
        t = tau.ustrip("s")
        return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")

    builder = cxfc.BishopBuilder(circle, "s")
    x = u.Q(jnp.asarray([2.0 * np.cos(1.0), 2.0 * np.sin(1.0), 0.0]), "km")
    tau = cxfc.nearest_tau(builder, x, bounds=(u.Q(0.0, "s"), u.Q(2 * np.pi, "s")))
    assert float(tau.ustrip("s")) == pytest.approx(1.0, abs=1e-3)


def test_a_degenerate_n_seed_is_refused_clearly() -> None:
    """`n_seed < 2` has no spacing to bracket around.

    Left unguarded it divides by zero and builds an empty grid, surfacing as a
    shape error from `argmin` that names nothing the caller did.
    """
    builder = cxfc.FrenetSerretBuilder(wiggly, "s")
    x = u.Q(jnp.asarray(PROBE), "km")
    with pytest.raises(ValueError, match="at least 2"):
        cxfc.nearest_tau(builder, x, bounds=BOUNDS, n_seed=1)
