"""A Bishop frame does not close around a closed curve, and the chart tears.

The tear is real geometry, not a solver defect: a rotation-minimising frame
carried once around a closed space curve comes back rotated about the tangent.
`TubularChart.holonomy` reports how far, so the seam is at least visible.

A **planar** closed curve has zero holonomy, so the obvious probe -- a circle --
measures nothing. Every test here that matters uses a trefoil.
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
    """The number must predict the gap, or it is just a number.

    Two points naming the same station and the same normal offset, one from
    each side of the seam: a frame that closed would put them together.
    """
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
