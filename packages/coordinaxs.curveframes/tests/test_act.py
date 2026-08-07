"""Tests for acting with curve-frame transforms on tangent data."""

import pytest

import coordinax.charts as cxc
import coordinax.representations as cxr
import coordinax.transforms as cxfm
import quaxed.numpy as jnp
import unxt as u

import coordinaxs.curveframes as cxfc


def _fs() -> cxfm.TimeDep:
    def circle(tau):
        t = tau.ustrip("s")
        return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.0 * t]), "km")

    return cxfm.TimeDep(cxfc.FrenetSerretBuilder(circle, u.unit("s")))


class TestTangentAnchorThreading:
    """Velocity transport must include the frame's own d/dtau motion.

    The transform is time-dependent, so the transported velocity is the
    *kinematic prolongation*, not the frozen-tau pushforward: it must match
    the total tau-derivative of the point action along the moving point.
    """

    def test_velocity_matches_finite_difference(self):
        op = _fs()
        tau = u.Q(0.3, "s")
        at = {"x": u.Q(2.0, "km"), "y": u.Q(1.0, "km"), "z": u.Q(0.5, "km")}
        v = {"x": u.Q(0.1, "km/s"), "y": u.Q(-0.2, "km/s"), "z": u.Q(0.05, "km/s")}

        got = cxfm.act(op, tau, v, cxc.cart3d, cxr.coord_vel, at=at)

        # Independent oracle: central difference of the point action along
        # the straight-line path through ``at`` with velocity ``v``.
        h = u.Q(1e-4, "s")

        def path(t):
            x = {k: at[k] + v[k] * (t - tau) for k in at}
            return cxfm.act(op, t, x, cxc.cart3d, cxr.point)

        plus, minus = path(tau + h), path(tau - h)
        for k in "xyz":
            fd = (plus[k] - minus[k]) / (2 * h)
            assert jnp.allclose(
                u.ustrip("km/s", got[k]), u.ustrip("km/s", fd), atol=1e-6
            )

    def test_velocity_is_not_the_frozen_pushforward(self):
        """The frame's own motion contributes: prolongation != pushforward."""
        op = _fs()
        tau = u.Q(0.3, "s")
        at = {"x": u.Q(2.0, "km"), "y": u.Q(1.0, "km"), "z": u.Q(0.5, "km")}
        v = {"x": u.Q(0.1, "km/s"), "y": u.Q(-0.2, "km/s"), "z": u.Q(0.05, "km/s")}

        prolonged = cxfm.act(op, tau, v, cxc.cart3d, cxr.coord_vel, at=at)
        frozen = cxfm.pushforward(op, tau, v, cxc.cart3d, cxr.coord_vel, at=at)
        assert not all(
            jnp.allclose(
                u.ustrip("km/s", prolonged[k]), u.ustrip("km/s", frozen[k]), atol=1e-6
            )
            for k in "xyz"
        )

    def test_velocity_without_at_raises(self):
        op = _fs()
        v = {"x": u.Q(0.1, "km/s"), "y": u.Q(-0.2, "km/s"), "z": u.Q(0.05, "km/s")}
        with pytest.raises(TypeError, match="requires the base point"):
            cxfm.act(op, u.Q(0.3, "s"), v, cxc.cart3d, cxr.coord_vel)
