"""Two-argument curves in the frame builders.

A builder is called with a *single* parameter. For a one-argument curve
that is the curve parameter; for a two-argument ``gamma(tau, t)`` it is
the time, and the station must then be pinned as a field. With neither
pinned, two unknowns face one slot, so construction raises.

With a station pinned, `builder(t)` is the frame of the time-t slice at
that station. A slice at fixed t is a one-argument curve -- what `AtTime`
produces -- so the existing machinery applies unchanged, and the
equivalence with the hand-built slice is the correctness claim these tests
turn on.

Before this, a two-argument curve constructed happily and then failed on
*every* call with a `TypeError` raised from inside the ODE solve, nowhere
near the construction that caused it.

The routing and the guard both live on `AbstractCurveFrameBuilder`, so
both concrete builders inherit them. Also pinned here are the two
one-argument idioms that must keep working: #712 found both misread when
arity was decided by counting parameters rather than checking whether the
second one is required.
"""

import functools as ft

import jax
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
def test_two_argument_curve_without_a_station_is_rejected(builder) -> None:
    """Both builders inherit the guard from `AbstractCurveFrameBuilder`."""
    with pytest.raises(ValueError, match="must be pinned"):
        builder(curve2, "km")


def test_the_error_names_the_remedy() -> None:
    """A message that only says "no" costs the reader the fix."""
    with pytest.raises(ValueError, match="AtTime"):
        cxfc.BishopBuilder(curve2, "km")


@pytest.mark.parametrize(
    "builder", [cxfc.BishopBuilder, cxfc.FrenetSerretBuilder], ids=["bishop", "frenet"]
)
def test_a_pinned_station_makes_the_call_time_parameter_the_time(builder) -> None:
    """The whole correctness claim: routing equals the hand-built slice.

    `builder(t)` on a two-argument curve must give the frame of the time-t
    slice at the pinned station -- which is what
    ``builder(AtTime(curve, t), ...)`` builds directly. Fails if `_resolve`
    binds the wrong slot, or slices at the wrong time.
    """
    s0 = u.Q(1.3, "km")
    b = builder(curve2, "km", gamma=s0)
    # t = 0 is excluded deliberately: `curve2(s, 0)` is the straight line
    # ``(s, 0, 0)``, where Frenet--Serret is singular (zero curvature, so the
    # normal is undefined and the rotation is NaN). That is the apparatus's
    # own degeneracy, not a routing question, and `BishopBuilder` exists
    # precisely because it does not have it.
    for t_val in (0.5, 1.7, 2.3):
        t = u.Q(t_val, "s")
        manual = builder(cxfc.AtTime(curve2, t), "km", gamma=s0)
        assert jnp.allclose(
            b.location(t).ustrip("km"), manual.location(t).ustrip("km"), atol=1e-12
        ), t_val
        assert jnp.allclose(
            b.tangent(t).ustrip(""), manual.tangent(t).ustrip(""), atol=1e-12
        ), t_val
        assert jnp.allclose(b.rotation_matrix(t), manual.rotation_matrix(t), atol=1e-12)


def test_the_frame_actually_evolves_with_time() -> None:
    """Guards against `_resolve` being an expensive no-op.

    A routing bug that ignored `t` would still satisfy the equivalence test
    above if the *manual* side ignored it too. This pins that the frame
    genuinely differs between two times -- on a curve that bends, so the
    rotation moves and not merely the origin.
    """
    b = cxfc.BishopBuilder(curve2, "km", gamma=u.Q(1.3, "km"))
    l0 = b.location(u.Q(0.0, "s")).ustrip("km")
    l1 = b.location(u.Q(1.7, "s")).ustrip("km")
    t0 = b.tangent(u.Q(0.0, "s")).ustrip("")
    t1 = b.tangent(u.Q(1.7, "s")).ustrip("")
    assert float(jnp.linalg.norm(l1 - l0)) > 1.0, (l0, l1)
    assert float(jnp.linalg.norm(t1 - t0)) > 0.1, (t0, t1)


def test_gradients_flow_to_time_and_to_the_station() -> None:
    """Both slots stay differentiable: `t` at call time, the station as a leaf.

    Checked against the closed form for
    ``gamma(s, t) = (s(1 + t/2), t s^2/10, 0)``:
    ``d(y)/dt = s^2/10`` and ``d(y)/ds = t s / 5``.
    """
    s0_val, t_val = 1.3, 1.0
    b = cxfc.BishopBuilder(curve2, "km", gamma=u.Q(s0_val, "km"))

    d_dt = jax.grad(lambda tv: b.location(u.Q(tv, "s")).ustrip("km")[1])(t_val)
    assert jnp.allclose(d_dt, 0.1 * s0_val**2, atol=1e-10), d_dt

    def loc_of_station(sv: float) -> float:
        moved = cxfc.BishopBuilder(curve2, "km", gamma=u.Q(sv, "km"))
        return moved.location(u.Q(t_val, "s")).ustrip("km")[1]

    d_ds = jax.grad(loc_of_station)(s0_val)
    assert jnp.allclose(d_ds, 0.2 * t_val * s0_val, atol=1e-10), d_ds


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
