r"""Degenerate inputs to `nearest_tau`: refuse promptly, never hang or mislead.

Two unrelated failure modes, both reachable through the public API.

A zero-width ``bounds`` used to hang forever: `Bisection`'s
``expand_if_necessary`` grows a bracket by *doubling its width*, doubling zero
never grows it, and that expansion is not bounded by ``max_steps``.

A finite ``s_max`` bounds where an `ArcLength` curve can be evaluated at all,
which interacts with `nearest_tau`'s documented out-of-bounds degradation:
that degradation returns a ``tau`` outside ``tau_bounds``, and such a ``tau``
needs interpolation coefficients too.
"""

import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc


def circle(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """Unit circle, radius 1 km."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")


PROBE = u.Q(jnp.asarray([2.0 * jnp.cos(1.0), 2.0 * jnp.sin(1.0), 0.0]), "km")
ARC_BOUNDS = (u.Q(0.0, "km"), u.Q(1.0, "km"))


def _at_angle(ang: float) -> u.AbstractQuantity:
    """A probe twice the circle's radius out, nearest the curve at ``tau=ang``."""
    return u.Q(2.0 * jnp.asarray([jnp.cos(ang), jnp.sin(ang), 0.0]), "km")


def test_zero_width_bounds_is_refused() -> None:
    """It must raise, and promptly -- hanging is the failure this guards."""
    builder = cxfc.BishopBuilder(circle, "s")
    with pytest.raises(ValueError, match="zero width"):
        cxfc.nearest_tau(builder, PROBE, bounds=(u.Q(3.0, "s"), u.Q(3.0, "s")))


def test_zero_width_bounds_is_refused_under_jit() -> None:
    """The eager path uses a Python branch; the traced path needs `error_if`."""
    builder = cxfc.BishopBuilder(circle, "s")

    @jax.jit
    def solve(lo: u.AbstractQuantity, hi: u.AbstractQuantity) -> u.AbstractQuantity:
        return cxfc.nearest_tau(builder, PROBE, bounds=(lo, hi))

    # `RuntimeError`, not `Exception`: `eqx.error_if` surfaces as
    # `JaxRuntimeError`, which subclasses it. Catching `Exception` would let an
    # unrelated tracing failure pass as the refusal under test.
    with pytest.raises(RuntimeError, match="zero width"):
        solve(u.Q(3.0, "s"), u.Q(3.0, "s"))


def test_a_proper_interval_still_works() -> None:
    """The guard must not disturb the ordinary case."""
    builder = cxfc.BishopBuilder(circle, "s")
    tau = cxfc.nearest_tau(builder, PROBE, bounds=(u.Q(0.0, "s"), u.Q(2 * jnp.pi, "s")))
    assert float(tau.ustrip("s")) == pytest.approx(1.0, abs=1e-3)


def test_s_max_does_not_disturb_an_in_bounds_query() -> None:
    """`s_max = tau_bounds[1]` is enough when the answer is inside the scan.

    This is exactly what `ArcLength.s_max` promises, and it holds.
    """
    free = cxfc.BishopBuilder(cxfc.ArcLength(circle, "s"), "km")
    pinned = cxfc.BishopBuilder(cxfc.ArcLength(circle, "s", s_max=u.Q(1.0, "km")), "km")
    x = _at_angle(0.5)

    want = float(cxfc.nearest_tau(free, x, bounds=ARC_BOUNDS).ustrip("km"))
    got = float(cxfc.nearest_tau(pinned, x, bounds=ARC_BOUNDS).ustrip("km"))

    assert got == pytest.approx(want, abs=1e-6)
    assert got == pytest.approx(0.5, abs=1e-3)


def test_s_max_must_cover_the_answer_not_just_the_scan() -> None:
    """An out-of-bounds answer needs coefficients too, so `s_max` must reach it.

    `nearest_tau` documents that a point nearest the curve *outside*
    ``tau_bounds`` degrades rather than raising -- but that needs the curve to
    be evaluable there. With ``s_max = tau_bounds[1] = 1 km`` the answer
    ``tau = 1.2`` lies past the interpolation, so it raises. Refusing is right:
    extrapolating off the end would return a plausible-looking wrong answer.
    Unset `s_max` and the same query degrades as documented.
    """
    x = _at_angle(1.2)

    free = cxfc.BishopBuilder(cxfc.ArcLength(circle, "s"), "km")
    assert float(cxfc.nearest_tau(free, x, bounds=ARC_BOUNDS).ustrip("km")) == (
        pytest.approx(1.2, abs=1e-3)
    )

    pinned = cxfc.BishopBuilder(cxfc.ArcLength(circle, "s", s_max=u.Q(1.0, "km")), "km")
    # `EquinoxRuntimeError` subclasses `RuntimeError`; this matches how
    # `test_arclength_smax.py` already pins the same guard.
    with pytest.raises(RuntimeError, match="outside the solved domain"):
        cxfc.nearest_tau(pinned, x, bounds=ARC_BOUNDS)
