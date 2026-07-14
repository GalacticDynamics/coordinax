"""Tests for parallel-transport transform act dispatches."""

import pytest

import quaxed.numpy as jnp
import unxt as u

import coordinax.charts as cxc
import coordinax.curveframes as cxfc
import coordinax.representations as cxr
import coordinax.transforms as cxfm


class TestTangentAnchorThreading:
    """The pipeline must advance anchors between its TD sub-transforms.

    Regression for the final-audit finding: forwarding `at` unchanged into
    the Rotate step evaluated dR/dtau at the un-translated base point.
    """

    @staticmethod
    def _fs():
        def circle(tau):
            t = tau.ustrip("s")
            return u.Q(jnp.stack([jnp.cos(t), jnp.sin(t), 0.0 * t]), "km")

        return cxfc.FrenetSerretTransform.from_curve(circle, tau_unit=u.unit("s"))

    def test_velocity_matches_generic_prolongation(self):
        from coordinax.transforms._src.actions.prolong import prolong_jet

        fs = self._fs()
        tau = u.Q(0.3, "s")
        at = {"x": u.Q(2.0, "km"), "y": u.Q(1.0, "km"), "z": u.Q(0.5, "km")}
        v = {"x": u.Q(0.1, "km/s"), "y": u.Q(-0.2, "km/s"), "z": u.Q(0.05, "km/s")}
        fast = cxfm.act(fs, tau, v, cxc.cart3d, cxr.coord_vel, at=at)
        gen = prolong_jet(fs, tau, {0: at, 1: v}, cxc.cart3d)
        for k in "xyz":
            assert jnp.allclose(
                u.ustrip("km/s", fast[k]), u.ustrip("km/s", gen[1][k]), atol=1e-6
            )

    def test_velocity_without_at_raises(self):
        fs = self._fs()
        v = {"x": u.Q(0.1, "km/s"), "y": u.Q(-0.2, "km/s"), "z": u.Q(0.05, "km/s")}
        with pytest.raises(TypeError, match="requires the base point"):
            cxfm.act(fs, u.Q(0.3, "s"), v, cxc.cart3d, cxr.coord_vel)
