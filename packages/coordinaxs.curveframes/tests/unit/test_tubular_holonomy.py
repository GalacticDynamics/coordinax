"""`TubularChart.holonomy` reports the seam a closed curve's frame tears at.

A planar closed curve has none, so the tests that matter use a trefoil.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import coordinax.charts as cxc
import unxt as u

import coordinaxs.curveframes as cxfc

TWO_PI = float(2 * np.pi)
CLOSED = (u.Q(0.0, "s"), u.Q(TWO_PI, "s"))
OPEN = (u.Q(0.0, "s"), u.Q(float(np.pi), "s"))


def trefoil(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    t = tau.ustrip("s")
    return u.Q(
        jnp.stack(
            [
                jnp.sin(t) + 2 * jnp.sin(2 * t),
                jnp.cos(t) - 2 * jnp.cos(2 * t),
                -jnp.sin(3 * t),
            ]
        ),
        "km",
    )


def planar_circle(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")


def _chart(curve, bounds):
    return cxfc.TubularChart(cxfc.BishopBuilder(curve, "s"), tau_bounds=bounds)


def _rad(ch) -> float:
    return float(ch.holonomy().ustrip("rad"))


def test_a_closed_space_curve_tears() -> None:
    """The trefoil's frame comes back -2.225041 rad round."""
    assert _rad(_chart(trefoil, CLOSED)) == pytest.approx(-2.225041, abs=1e-5)


def test_a_planar_closed_curve_does_not() -> None:
    """Why this is easy to miss: the obvious probe reads exactly zero."""
    assert _rad(_chart(planar_circle, CLOSED)) == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("curve", [trefoil, planar_circle])
def test_an_open_curve_has_no_holonomy(curve) -> None:
    """Holonomy belongs to a loop. No loop, no seam, nothing to report."""
    assert _rad(_chart(curve, OPEN)) == pytest.approx(0.0, abs=1e-8)


def test_the_holonomy_is_the_size_of_the_actual_tear() -> None:
    """The angle must predict the gap, or it is just a number."""
    ch = _chart(trefoil, CLOSED)
    eps = 1e-7
    off = {"n1": u.Q(0.2, "km"), "n2": u.Q(0.0, "km")}
    ends = [
        cxc.pt_map({"tau": u.Q(t, "s"), **off}, ch.M, ch, ch.M, cxc.cart3d)
        for t in (eps, TWO_PI - eps)
    ]
    a, b = (jnp.stack([e[k].ustrip("km") for k in ("x", "y", "z")]) for e in ends)
    gap = float(jnp.linalg.norm(a - b))

    # A pure rotation by `holonomy` about the tangent moves a point at radius
    # `r` by `2 r |sin(theta/2)|`. That the measured gap matches is what says
    # the reported angle *is* the tear.
    r = 0.2
    assert gap == pytest.approx(2 * r * abs(np.sin(_rad(ch) / 2)), rel=1e-3)
    assert gap == pytest.approx(0.358726, abs=1e-5)


def test_a_planar_seam_actually_closes() -> None:
    """The control: zero holonomy must mean the two sides coincide."""
    ch = _chart(planar_circle, CLOSED)
    eps = 1e-7
    off = {"n1": u.Q(0.2, "km"), "n2": u.Q(0.0, "km")}
    ends = [
        cxc.pt_map({"tau": u.Q(t, "s"), **off}, ch.M, ch, ch.M, cxc.cart3d)
        for t in (eps, TWO_PI - eps)
    ]
    a, b = (jnp.stack([e[k].ustrip("km") for k in ("x", "y", "z")]) for e in ends)
    assert float(jnp.linalg.norm(a - b)) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("n_scale", [1, 2])
def test_too_few_scale_samples_are_refused(n_scale: int) -> None:
    """`n_scale=2` missed a real seam; `n_scale=1` invented one on an open curve."""
    with pytest.raises(ValueError, match="at least 3"):
        _chart(trefoil, CLOSED).holonomy(n_scale=n_scale)


@pytest.mark.parametrize("n_scale", [3, 4, 8, 32])
def test_the_answer_does_not_depend_on_the_scale_sampling(n_scale: int) -> None:
    """Above the floor it is a size estimate, and the verdict is insensitive."""
    assert _rad_at(_chart(trefoil, CLOSED), n_scale) == pytest.approx(
        -2.225041, abs=1e-5
    )
    assert _rad_at(_chart(trefoil, OPEN), n_scale) == pytest.approx(0.0, abs=1e-8)


def _rad_at(ch, n_scale: int) -> float:
    return float(ch.holonomy(n_scale=n_scale).ustrip("rad"))


def test_an_open_curve_never_evaluates_the_frame() -> None:
    """A straight line has no normal, but no seam either: 0, not a raise."""

    def line(tau: u.AbstractQuantity) -> u.AbstractQuantity:
        t = tau.ustrip("s")
        return u.Q(jnp.stack([t, jnp.zeros_like(t), jnp.zeros_like(t)]), "km")

    ch = cxfc.TubularChart(
        cxfc.FrenetSerretBuilder(line, "s"), tau_bounds=(u.Q(0.0, "s"), u.Q(1.0, "s"))
    )
    assert float(ch.holonomy().ustrip("rad")) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("bounds", [CLOSED, OPEN])
def test_jit_agrees_with_eager(bounds) -> None:
    """The branch is a `lax.cond` once traced and a Python `if` when not."""
    ch = _chart(trefoil, bounds)
    import jax

    assert float(jax.jit(ch.holonomy)().ustrip("rad")) == pytest.approx(
        float(ch.holonomy().ustrip("rad")), abs=1e-9
    )
