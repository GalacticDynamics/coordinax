"""Tests for the ``embedded_twosphere`` ambient-chart selection."""

__all__: tuple[str, ...] = ()

import jax.numpy as jnp
import numpy as np
import pytest

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


@pytest.fixture(scope="module")
def unit_sphere():
    """The unit two-sphere embedded in R^3 (dimensionless radius)."""
    return cxm.EmbeddedManifold(
        intrinsic=cxm.S2, ambient=cxm.R3, embed_map=cxm.TwoSphereIn3D(radius=1)
    )


# Induced metric of the unit two-sphere, as textbook closed forms. Deriving
# these independently of the J^T J machinery is the whole point: an R^3
# embedding reference would share its method with the code under test.
#   sph2      ds2 = dtheta^2 + sin^2(theta) dphi^2
#   lonlat    ds2 = cos^2(lat) dlon^2 + dlat^2
#   math      as sph2 with (theta, phi) = (azimuth, polar) swapped
#   loncoslat substituting lon = x / cos(y) into the lonlat form gives
#             ds2 = dx^2 + 2 x tan(y) dx dy + (1 + x^2 tan^2(y)) dy^2
_METRIC_CASES = [
    (cxc.sph2, {"theta": 0.9, "phi": 0.6}, np.diag([1.0, np.sin(0.9) ** 2])),
    (cxc.lonlat_sph2, {"lon": 0.6, "lat": 0.7}, np.diag([np.cos(0.7) ** 2, 1.0])),
    (cxc.math_sph2, {"theta": 0.6, "phi": 0.9}, np.diag([np.sin(0.9) ** 2, 1.0])),
    (
        cxc.loncoslat_sph2,
        {"lon_coslat": 0.3, "lat": 0.5},
        np.array(
            [
                [1.0, 0.3 * np.tan(0.5)],
                [0.3 * np.tan(0.5), 1.0 + 0.3**2 * np.tan(0.5) ** 2],
            ]
        ),
    ),
]


class TestEmbeddedMetricRespectsChart:
    """metric_matrix returns the induced metric in the *passed* chart's coords."""

    @pytest.mark.parametrize(("chart", "coords", "expected"), _METRIC_CASES)
    def test_metric_matches_closed_form(self, unit_sphere, chart, coords, expected):
        pt = {k: u.Q(v, "rad") for k, v in coords.items()}
        g = cxm.metric_matrix(unit_sphere, pt, chart)
        # rtol=0: the check is purely absolute, else assert_allclose's default
        # rtol=1e-7 would swamp the 1e-12 atol.
        np.testing.assert_allclose(
            np.asarray(g.matrix.value), expected, rtol=0, atol=1e-12
        )
        # Units: dimensionless ambient (radius=1) over angular coords, so
        # every g_ij is cart_unit^2 / (rad * rad) = 1 / rad^2.
        unit = g.matrix.unit
        assert all(str(unit[i, j]) == "1 / rad2" for i in range(2) for j in range(2))

    def test_unsupported_chart_is_rejected(self, unit_sphere) -> None:
        """A chart outside the intrinsic atlas raises, naming the atlas."""
        pt = {"x": u.Q(0.3, ""), "y": u.Q(0.4, "")}
        with pytest.raises(ValueError, match="not supported by the manifold"):
            cxm.metric_matrix(unit_sphere, pt, cxc.cart2d)

    @pytest.mark.parametrize("batch", [(3,), (2, 3)])
    def test_metric_is_batch_safe(self, unit_sphere, batch) -> None:
        """A batched point gives (*batch, n, n) matching the per-point metrics."""
        theta = jnp.linspace(0.2, 1.3, np.prod(batch)).reshape(batch)
        pt = {"theta": u.Q(theta, "rad"), "phi": u.Q(theta / 2, "rad")}
        m = jnp.asarray(cxm.metric_matrix(unit_sphere, pt, cxc.sph2).matrix.value)
        assert m.shape == (*batch, 2, 2)
        for idx in np.ndindex(batch):
            single = {k: u.Q(v.value[idx], u.unit_of(v)) for k, v in pt.items()}
            ref = cxm.metric_matrix(unit_sphere, single, cxc.sph2).matrix.value
            assert jnp.allclose(m[idx], jnp.asarray(ref))
