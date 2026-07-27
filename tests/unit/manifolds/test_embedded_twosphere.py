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

    def test_usys_is_forwarded_to_embed(self) -> None:
        """``pt_embed`` accepts and threads ``usys`` to the embedding.

        Bare-array intrinsic coords embed through a Cartesian ambient.
        """
        m = cxm.embedded_twosphere(radius=u.Q(2.0, "m"), ambient=cxc.cart3d)
        p = {"theta": jnp.asarray(jnp.pi / 2), "phi": jnp.asarray(0.0)}
        out = cxm.pt_embed(p, m, usys=u.unitsystems.si)
        assert set(out) == {"x", "y", "z"}
        np.testing.assert_allclose(np.asarray(out["x"]), 2.0, atol=1e-6)

    def test_non_singleton_spherical_ambient_takes_spherical_path(self) -> None:
        """A non-singleton Spherical3D ambient still embeds to spherical coords.

        The ambient check is by type, not identity, so a freshly-constructed
        ``Spherical3D`` (distinct from the ``sph3d`` instance) is handled too.
        """
        fresh = type(cxc.sph3d)(M=cxc.sph3d.M)
        assert fresh is not cxc.sph3d  # a distinct instance
        m = cxm.embedded_twosphere(radius=u.Q(2.0, "km"), ambient=fresh)
        out = cxm.pt_embed(_P, m)
        assert set(out) == {"r", "theta", "phi"}
        np.testing.assert_allclose(u.ustrip("km", out["r"]), 2.0, atol=1e-6)


class TestEmbeddedMetricRespectsChart:
    """metric_matrix returns the induced metric in the *passed* chart's coords."""

    @staticmethod
    def _embedding_metric(chart, coords):
        """J^T J of the unit-sphere R^3 embedding, expressed in ``chart`` coords."""
        import jax

        keys = chart.components

        def embed(x):
            p = {k: u.Q(x[i], "rad") for i, k in enumerate(keys)}
            s = cxc.pt_map(p, chart, cxc.sph2)
            th, ph = u.ustrip("rad", s["theta"]), u.ustrip("rad", s["phi"])
            return jnp.array(
                [jnp.sin(th) * jnp.cos(ph), jnp.sin(th) * jnp.sin(ph), jnp.cos(th)]
            )

        x0 = jnp.array([coords[k] for k in keys])
        j = jax.jacfwd(embed)(x0)
        return np.asarray(j.T @ j)

    def test_metric_in_each_two_sphere_chart(self) -> None:
        emb = cxm.EmbeddedManifold(
            intrinsic=cxm.S2, ambient=cxm.R3, embed_map=cxm.TwoSphereIn3D(radius=1)
        )
        cases = [
            (cxc.sph2, {"theta": 0.9, "phi": 0.6}),
            (cxc.lonlat_sph2, {"lon": 0.6, "lat": 0.7}),
            (cxc.loncoslat_sph2, {"lon_coslat": 0.3, "lat": 0.5}),
            (cxc.math_sph2, {"theta": 0.6, "phi": 0.9}),
        ]
        for chart, coords in cases:
            pt = {k: u.Q(v, "rad") for k, v in coords.items()}
            g = cxm.metric_matrix(emb, pt, chart)
            got = np.asarray(g.matrix.value)
            ref = self._embedding_metric(chart, coords)
            assert np.allclose(got, ref, atol=1e-9), f"{chart}: {got} != {ref}"
