"""Builders reject a two-argument curve at construction.

A builder is called with a single parameter, so a curve needing
``(tau, t)`` leaves the time unbound. Before this guard such a curve
constructed happily and then failed on *every* call, with a `TypeError`
raised from inside the ODE solve -- nowhere near the construction that
caused it.

The guard lives on `AbstractCurveFrameBuilder.__check_init__`, so both
concrete builders inherit it; the tests below pin that, and pin the two
one-argument idioms that must keep working (#712 found both of these
misread when arity was decided by counting parameters rather than
checking whether the second one is required).
"""

import functools as ft

import jax.numpy as jnp
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc


def curve1(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    """An ordinary one-argument curve."""
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.3 * t]), "km")


def curve2(s: u.AbstractQuantity, t: u.AbstractQuantity) -> u.AbstractQuantity:
    """A time-dependent curve: it bends and stretches with ``t``."""
    sv, tv = s.ustrip("km"), t.ustrip("s")
    x = sv * (1.0 + 0.5 * tv)
    y = 0.1 * tv * sv**2
    return u.Q(jnp.stack([x, y, jnp.zeros_like(x)]), "km")


def curve_with_knob(
    tau: u.AbstractQuantity, smoothing: float = 0.1
) -> u.AbstractQuantity:
    """One-argument, but carrying a tuning knob with a default."""
    del smoothing
    return curve1(tau)


@pytest.mark.parametrize(
    "builder", [cxfc.BishopBuilder, cxfc.FrenetSerretBuilder], ids=["bishop", "frenet"]
)
def test_two_argument_curve_is_rejected_at_construction(builder) -> None:
    """Both builders inherit the guard from `AbstractCurveFrameBuilder`."""
    with pytest.raises(ValueError, match="one-argument curve"):
        builder(curve2, "km")


def test_the_error_names_the_remedy() -> None:
    """A message that only says "no" costs the reader the fix."""
    with pytest.raises(ValueError, match="AtTime"):
        cxfc.BishopBuilder(curve2, "km")


def test_a_pinned_gamma_does_not_excuse_a_two_argument_curve() -> None:
    """`gamma` fixes the station, not the time -- the curve is still unusable.

    Fails if the guard is ever narrowed to ``gamma is None`` before the
    routing that would make a pinned two-argument curve actually work.
    """
    with pytest.raises(ValueError, match="one-argument curve"):
        cxfc.BishopBuilder(curve2, "km", gamma=u.Q(1.3, "km"))


def test_at_time_makes_a_two_argument_curve_usable() -> None:
    """The remedy the message names has to actually work."""
    frozen = cxfc.AtTime(curve2, u.Q(0.5, "s"))
    b = cxfc.BishopBuilder(frozen, "km")
    # gamma(s=1.3, t=0.5) = (1.3 * 1.25, 0.1 * 0.5 * 1.69, 0)
    got = b.location(u.Q(1.3, "km")).ustrip("km")
    assert jnp.allclose(got, jnp.array([1.625, 0.0845, 0.0]), atol=1e-8), got


def test_a_defaulted_second_parameter_is_still_one_argument() -> None:
    """``def curve(tau, smoothing=0.1)`` is a one-argument curve."""
    cxfc.BishopBuilder(curve_with_knob, "s")


def test_a_partial_frozen_time_is_still_one_argument() -> None:
    """`ft.partial` leaves the bound parameter visible, with a default."""
    frozen = ft.partial(curve2, t=u.Q(0.5, "s"))
    cxfc.BishopBuilder(frozen, "km")
