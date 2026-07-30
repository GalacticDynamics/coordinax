"""Tests for the scale_factors() manifold API and wrappers."""

import jax
import jax.numpy as jnp
import pytest

import unxt as u
import unxts.linalg as ul

import coordinax.charts as cxc
import coordinax.manifolds as cxm
from coordinax._src.metric.matrix import DiagonalMetric
from coordinaxs.api.manifolds import metric_matrix as mm_dispatch


class TestScaleFactorsEuclidean:
    """Tests for scale_factors on Euclidean metrics and manifolds."""

    def test_cartesian_metric_returns_1d_QuantityMatrix(self):
        metric = cxm.FlatMetric(3)
        at = {
            "x": u.Q(jnp.array(1), "m"),
            "y": u.Q(jnp.array(2), "m"),
            "z": u.Q(jnp.array(3), "m"),
        }

        result = cxm.scale_factors(metric, cxc.cart3d, at=at)

        assert isinstance(result, ul.QuantityMatrix)
        assert result.shape == (3,)
        assert result.ndim == 1
        assert jnp.allclose(result.value, jnp.array([1, 1, 1]))
        assert all(result.unit[i] == u.unit("") for i in range(3))

    def test_spherical_metric_returns_metric_diagonal_entries(self):
        metric = cxm.FlatMetric(3)
        at = {
            "r": u.Q(jnp.array(2), "m"),
            "theta": u.Angle(jnp.pi / 6, "rad"),
            "phi": u.Angle(jnp.array(0.4), "rad"),
        }

        result = cxm.scale_factors(metric, cxc.sph3d, at=at)

        assert isinstance(result, ul.QuantityMatrix)
        assert result.shape == (3,)
        assert jnp.allclose(result.value, jnp.array([1, 4, 1]), atol=1e-6)
        assert result.unit[0] == u.unit("")
        assert result.unit[1] == u.unit("m2 / rad2")
        assert result.unit[2] == u.unit("m2 / rad2")


class TestScaleFactorsGeneric:
    """Tests for generic metric-based scale_factors behavior."""

    def test_hyperspherical_bare_arrays_promote_to_QuantityMatrix(self):
        metric = cxm.RoundMetric(ndim=2)
        at = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0)}

        result = cxm.scale_factors(metric, cxc.sph2, at=at)

        assert isinstance(result, ul.QuantityMatrix)
        assert result.shape == (2,)
        assert jnp.allclose(result.value, jnp.array([1, 1]), atol=1e-6)
        assert all(result.unit[i] == u.unit("") for i in range(2))

    def test_generic_path_matches_metric_matrix_diag(self):
        metric = cxm.RoundMetric(ndim=2)
        at = {
            "theta": u.Angle(jnp.pi / 3, "rad"),
            "phi": u.Angle(jnp.array(0.1), "rad"),
        }

        # S2 in sph2 returns DiagonalMetric; diagonal IS the scale factors
        expected_mm = mm_dispatch(cxm.HyperSphericalManifold(2), at, cxc.sph2)
        assert isinstance(expected_mm, DiagonalMetric)
        # Extract numeric diagonal values
        diag = expected_mm.diagonal
        expected_values = diag.value if isinstance(diag, ul.QuantityMatrix) else diag

        result = cxm.scale_factors(metric, cxc.sph2, at=at)

        assert isinstance(result, ul.QuantityMatrix)
        assert jnp.allclose(result.value, expected_values, atol=1e-6)

    def test_jit(self):
        metric = cxm.RoundMetric(ndim=2)

        @jax.jit
        def compute(at):
            return cxm.scale_factors(metric, cxc.sph2, at=at)

        at = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(jnp.array(0), "rad")}
        result = compute(at)

        assert isinstance(result, ul.QuantityMatrix)
        assert jnp.allclose(result.value, jnp.array([1, 1]), atol=1e-6)

    def test_vmap_values(self):
        metric = cxm.RoundMetric(ndim=2)
        thetas = jnp.array([jnp.pi / 6, jnp.pi / 4, jnp.pi / 2])

        def compute(theta):
            return cxm.scale_factors(
                metric, cxc.sph2, at={"theta": theta, "phi": jnp.array(0)}
            ).value

        result = jax.vmap(compute)(thetas)
        expected = jnp.stack([jnp.ones_like(thetas), jnp.sin(thetas) ** 2], axis=-1)
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_embedded_manifold_requires_induced_metric(self):
        M = cxm.EmbeddedManifold(
            intrinsic=cxm.S2,
            ambient=cxm.R3,
            embed_map=cxm.TwoSphereIn3D(radius=u.Q(jnp.array(2), "m")),
        )
        assert isinstance(M.metric, cxm.PullbackMetric)

        at = {"theta": u.Angle(jnp.pi / 6, "rad"), "phi": u.Angle(jnp.array(0), "rad")}

        result = cxm.scale_factors(M.metric, cxc.sph2, at=at)

        # A 2-sphere of radius R embedded in Euclidean 3-space has induced metric
        # diag(R^2, R^2 sin^2(theta)) in the (theta, phi) chart. Here R = 2 m,
        # so the first diagonal entry is always 4 m^2 / rad^2.
        #
        # At theta = pi/6, sin^2(theta) = 1/4, so the second diagonal entry is
        # 4 * 1/4 = 1 with the same units. Using a non-equatorial point makes it
        # clear that we are testing the induced metric of the embedded sphere,
        # not just a coincidental [4, 4] value at the equator.
        expected = jnp.array([4, 1])

        assert isinstance(result, ul.QuantityMatrix)
        assert result.shape == (2,)
        assert jnp.allclose(result.value, expected, atol=1e-6)
        assert result.unit[0] == u.unit("m2 / rad2")
        assert result.unit[1] == u.unit("m2 / rad2")


class TestScaleFactorsPullbackMetric:
    """`scale_factors` on a `PullbackMetric` must agree with `metric_matrix`.

    It used to read the point with ``embed_map.intrinsic.components`` and ignore
    the passed ``chart`` -- the same defect #604 fixed for ``metric_matrix``.
    It also dropped the ambient Gram and was not batch-safe.
    """

    @staticmethod
    def _sphere(radius=u.Q(2.0, "m")):
        return cxm.EmbeddedManifold(
            intrinsic=cxm.S2, ambient=cxm.R3, embed_map=cxm.TwoSphereIn3D(radius=radius)
        )

    @pytest.mark.parametrize(
        ("chart", "at"),
        [
            (cxc.sph2, {"theta": jnp.pi / 2, "phi": 0.0}),
            (cxc.lonlat_sph2, {"lon": 0.0, "lat": 0.0}),
            (cxc.math_sph2, {"theta": 0.0, "phi": jnp.pi / 2}),
            (cxc.loncoslat_sph2, {"lon_coslat": 0.0, "lat": 0.0}),
        ],
    )
    def test_matches_metric_matrix_diagonal(self, chart, at):
        """Any valid intrinsic chart, not just the embedding's own."""
        M = self._sphere()
        pt = {k: u.Angle(v, "rad") for k, v in at.items()}
        h = jnp.asarray(cxm.scale_factors(M.metric, chart, at=pt).value)
        g = jnp.asarray(mm_dispatch(M, pt, chart).matrix.value)
        assert jnp.allclose(h, jnp.diagonal(g, axis1=-2, axis2=-1), atol=1e-6)
        # Every point above is on the equator of a radius-2 sphere -> 4 m2/rad2.
        assert jnp.allclose(h, jnp.asarray([4.0, 4.0]), atol=1e-5)

    def test_units_match_metric_matrix(self):
        M = self._sphere()
        pt = {"lon": u.Angle(0.0, "rad"), "lat": u.Angle(0.0, "rad")}
        h = cxm.scale_factors(M.metric, cxc.lonlat_sph2, at=pt)
        g = mm_dispatch(M, pt, cxc.lonlat_sph2)
        assert h.unit[0] == g.matrix.unit[0, 0]

    def test_ambient_metric_the_route_cannot_apply_is_refused(self):
        """A hand-built pullback can name an ambient metric this route can't use.

        The delegate evaluates the Gram on the ambient *manifold*, so only
        ``embed_map.ambient.M.metric`` is reachable. Returning the diagonal of
        that other metric instead would be the silent-wrong-answer this class
        exists to guard against.
        """
        pb = cxm.PullbackMetric(
            cxm.TwoSphereIn3D(radius=u.Q(2.0, "m")), cxm.RoundMetric(3)
        )
        at = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
        with pytest.raises(NotImplementedError, match="cannot be evaluated"):
            cxm.scale_factors(pb, cxc.sph2, at=at)

        # An `EmbeddedManifold`'s own metric always agrees, so it never trips.
        assert self._sphere().metric.ambient_metric == cxc.sph3d.M.metric

    def test_batched_point(self):
        """A batched point returns (*batch, n), not a shape error."""
        M = self._sphere()
        pt = {
            "lon": u.Angle(jnp.array([0.0, 0.3, 0.6]), "rad"),
            "lat": u.Angle(jnp.array([0.0, 0.2, 0.4]), "rad"),
        }
        h = jnp.asarray(cxm.scale_factors(M.metric, cxc.lonlat_sph2, at=pt).value)
        assert h.shape == (3, 2)
        for i in range(3):
            hi = cxm.scale_factors(
                M.metric, cxc.lonlat_sph2, at={k: v[i] for k, v in pt.items()}
            )
            assert jnp.allclose(h[i], jnp.asarray(hi.value), atol=1e-6)

    @pytest.mark.parametrize(
        ("name", "ct_of", "x_of", "expected"),
        [
            ("timelike", lambda s: s, lambda s: s * 0, -1.0),
            ("spacelike", lambda s: s * 0, lambda s: s, 1.0),
            ("null", lambda s: s, lambda s: s, 0.0),
        ],
        ids=lambda v: v if isinstance(v, str) else "",
    )
    def test_lorentzian_ambient_keeps_the_signature(self, name, ct_of, x_of, expected):
        """Squared column norms are always positive; the ambient Gram is not."""
        del name
        embed_map = cxm.CustomEmbeddingMap(
            intrinsic=cxc.cart1d,
            ambient=cxc.minkowskict,
            embed_fn=lambda p, *, usys=None: {
                "ct": ct_of(p["x"]),
                "x": x_of(p["x"]),
                "y": p["x"] * 0,
                "z": p["x"] * 0,
            },
            project_fn=lambda p, *, usys=None: {"x": p["ct"]},
        )
        M = cxm.EmbeddedManifold(
            intrinsic=cxm.R1, ambient=cxm.minkowski4d, embed_map=embed_map
        )
        h = cxm.scale_factors(M.metric, cxc.cart1d, at={"x": u.Q(1.0, "m")})
        assert jnp.allclose(jnp.asarray(h.value).ravel()[0], expected, atol=1e-9)
