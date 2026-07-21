"""Tests for the ``embedded_twosphere`` ambient-chart selection."""

__all__: tuple[str, ...] = ()

import jax.numpy as jnp
import numpy as np

import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm

_P = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}


class TestEmbeddedTwosphereAmbient:
    """The ``ambient`` argument selects the ambient coordinate chart."""

    def test_default_ambient_is_spherical(self) -> None:
        """Default ambient embeds to spherical ``(r, theta, phi)``."""
        m = cxm.embedded_twosphere(radius=u.Q(2.0, "km"))
        out = cxm.pt_embed(_P, m)
        assert set(out) == {"r", "theta", "phi"}
        np.testing.assert_allclose(u.ustrip("km", out["r"]), 2.0, atol=1e-6)

    def test_cartesian_ambient_embeds_to_xyz(self) -> None:
        """``ambient=cart3d`` embeds to Cartesian ``(x, y, z)``."""
        m = cxm.embedded_twosphere(radius=u.Q(2.0, "km"), ambient=cxc.cart3d)
        out = cxm.pt_embed(_P, m)
        # (r=2, theta=pi/2, phi=0) -> (x=2, y=0, z=0)
        assert set(out) == {"x", "y", "z"}
        np.testing.assert_allclose(u.ustrip("km", out["x"]), 2.0, atol=1e-6)
        np.testing.assert_allclose(u.ustrip("km", out["y"]), 0.0, atol=1e-6)
        np.testing.assert_allclose(u.ustrip("km", out["z"]), 0.0, atol=1e-6)

    def test_cartesian_ambient_roundtrip(self) -> None:
        """Embed then project through a Cartesian ambient recovers the point."""
        m = cxm.embedded_twosphere(radius=u.Q(2.0, "km"), ambient=cxc.cart3d)
        back = cxm.pt_project(cxm.pt_embed(_P, m), m)
        np.testing.assert_allclose(
            u.ustrip("rad", back["theta"]), u.ustrip("rad", _P["theta"]), atol=1e-6
        )
        np.testing.assert_allclose(
            u.ustrip("rad", back["phi"]), u.ustrip("rad", _P["phi"]), atol=1e-6
        )
