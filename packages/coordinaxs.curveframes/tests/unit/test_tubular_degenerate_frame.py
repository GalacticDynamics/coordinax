"""A transversely-moving worldtube is not a reach failure, and says so.

`jacobian_factor` vanishing has two causes that want different words. Past the
*focal distance* the offset cancels the local curvature: the tube axis is
healthy and pulling `n` inward fixes it. On a worldtube whose station moves with
no component along its own spatial tangent, `dx/dtau` lies in `span(U1, U2)` and
the axis itself is singular -- no offset helps, and naming a reach sends the
reader to adjust the one thing that cannot matter.

The guard also has to *fire*. It compares against `sqrt(eps)` rather than `0`
because an exactly degenerate chart does not come back exactly zero: the rod
below reads `0.0` eagerly and `1.9469e-17` under `jit`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import unxt as u

import coordinaxs.curveframes as cxfc

S0 = 1.3
TIME_BOUNDS = (u.Q(0.0, "s"), u.Q(2.0, "s"))
CIRCLE_BOUNDS = (u.Q(0.0, "s"), u.Q(float(2 * np.pi), "s"))


def circle(tau: u.AbstractQuantity) -> u.AbstractQuantity:
    t = tau.ustrip("s")
    return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), jnp.zeros_like(t)]), "km")


def rotating_rod(s: u.AbstractQuantity, t: u.AbstractQuantity) -> u.AbstractQuantity:
    """A rigid rod spun about its end: the station's velocity is transverse."""
    sv, tv = s.ustrip("km"), t.ustrip("s")
    return u.Q(
        jnp.stack([sv * jnp.cos(tv), sv * jnp.sin(tv), jnp.zeros_like(sv * tv)]), "km"
    )


def _rod():
    return cxfc.TubularChart(
        cxfc.BishopBuilder(rotating_rod, "km", station=u.Q(S0, "km"), normal_0="auto"),
        tau_bounds=TIME_BOUNDS,
    )


def _circle():
    return cxfc.TubularChart(
        cxfc.BishopBuilder(circle, "s", normal_0="auto"), tau_bounds=CIRCLE_BOUNDS
    )


def _at(n1: float, n2: float) -> dict:
    return {"tau": u.Q(0.7, "s"), "n1": u.Q(n1, "km"), "n2": u.Q(n2, "km")}


@pytest.mark.parametrize("n1", [0.0, 0.2, 0.5])
def test_the_whole_n2_zero_plane_is_singular(n1: float) -> None:
    """Including the axis itself, and at every offset along `n1`."""
    assert float(_rod().jacobian_factor(_at(n1, 0.0))) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("n1", "n2", "want"), [(0.0, 0.1, 0.1), (0.2, 0.1, 0.1), (0.5, 0.5, 0.5)]
)
def test_off_that_plane_the_factor_is_n2_over_the_station(n1, n2, want) -> None:
    """It is not degenerate everywhere -- an earlier draft of the spec said so."""
    assert float(_rod().jacobian_factor(_at(n1, n2))) == pytest.approx(
        want / S0, rel=1e-6
    )


def test_the_message_names_the_frame_not_the_reach() -> None:
    with pytest.raises(ValueError, match="degenerate at the tube axis"):
        _rod().check_data(_at(0.2, 0.0), values=True)


def test_a_real_focal_failure_still_names_the_reach() -> None:
    """The discriminator must not swallow the case it was built beside.

    Past the focal distance the axis stays healthy -- measured 1.0 on the unit
    circle while `n1=-1.6` reads negative.
    """
    ch = _circle()
    assert float(ch.jacobian_factor(_at(0.0, 0.0))) == pytest.approx(1.0, rel=1e-6)
    with pytest.raises(ValueError, match="outside the reach"):
        ch.check_data(_at(-1.6, 0.0), values=True)


@pytest.mark.parametrize(
    ("chart", "n1", "n2", "needle"),
    [
        (_rod, 0.2, 0.0, "degenerate at the tube axis"),
        (_circle, -1.6, 0.0, "outside the reach"),
    ],
)
def test_jit_fires_the_same_diagnosis_as_eager(chart, n1, n2, needle) -> None:
    """The exactly-degenerate case used to pass under `jit` and raise eagerly.

    `jacobian_factor` returns `1.9469e-17` there rather than `0.0`, so a bare
    `> 0` held. Both paths must now refuse, with the same words.
    """
    ch = chart()
    at = _at(n1, n2)
    with pytest.raises(ValueError, match=needle):
        ch.check_data(at, values=True)
    with pytest.raises(RuntimeError, match=needle):
        jax.jit(lambda d: ch.check_data(d, values=True))(at)


@pytest.mark.parametrize(("chart", "n1", "n2"), [(_rod, 0.2, 0.1), (_circle, 0.2, 0.0)])
def test_healthy_points_still_pass_both_ways(chart, n1, n2) -> None:
    """The `sqrt(eps)` floor must not start refusing ordinary offsets."""
    ch = chart()
    at = _at(n1, n2)
    ch.check_data(at, values=True)
    jax.jit(lambda d: ch.check_data(d, values=True))(at)
