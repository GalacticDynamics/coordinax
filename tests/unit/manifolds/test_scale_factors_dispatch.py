"""Tests for the scale_factors() manifold API and wrappers."""

import jax
import jax.numpy as jnp

import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm
from coordinax._src.metric.matrix import DiagonalMetric
from coordinax.api.manifolds import metric_matrix as mm_dispatch
from coordinax.internal import QMatrix


class TestScaleFactorsEuclidean:
    """Tests for scale_factors on Euclidean metrics and manifolds."""

    def test_cartesian_metric_returns_1d_QMatrix(self):
        metric = cxm.FlatMetric(3)
        at = {
            "x": u.Q(jnp.array(1), "m"),
            "y": u.Q(jnp.array(2), "m"),
            "z": u.Q(jnp.array(3), "m"),
        }

        result = cxm.scale_factors(metric, cxc.cart3d, at=at)

        assert isinstance(result, QMatrix)
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

        assert isinstance(result, QMatrix)
        assert result.shape == (3,)
        assert jnp.allclose(result.value, jnp.array([1, 4, 1]), atol=1e-6)
        assert result.unit[0] == u.unit("")
        assert result.unit[1] == u.unit("m2 / rad2")
        assert result.unit[2] == u.unit("m2 / rad2")


class TestScaleFactorsGeneric:
    """Tests for generic metric-based scale_factors behavior."""

    def test_hyperspherical_bare_arrays_promote_to_QMatrix(self):
        metric = cxm.RoundMetric(ndim=2)
        at = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0)}

        result = cxm.scale_factors(metric, cxc.sph2, at=at)

        assert isinstance(result, QMatrix)
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
        expected_values = diag.value if isinstance(diag, QMatrix) else diag

        result = cxm.scale_factors(metric, cxc.sph2, at=at)

        assert isinstance(result, QMatrix)
        assert jnp.allclose(result.value, expected_values, atol=1e-6)

    def test_jit(self):
        metric = cxm.RoundMetric(ndim=2)

        @jax.jit
        def compute(at):
            return cxm.scale_factors(metric, cxc.sph2, at=at)

        at = {
            "theta": u.Angle(jnp.pi / 2, "rad"),
            "phi": u.Angle(jnp.array(0), "rad"),
        }
        result = compute(at)

        assert isinstance(result, QMatrix)
        assert jnp.allclose(result.value, jnp.array([1, 1]), atol=1e-6)

    def test_vmap_values(self):
        metric = cxm.RoundMetric(ndim=2)
        thetas = jnp.array([jnp.pi / 6, jnp.pi / 4, jnp.pi / 2])

        def compute(theta):
            return cxm.scale_factors(
                metric,
                cxc.sph2,
                at={"theta": theta, "phi": jnp.array(0)},
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

        at = {
            "theta": u.Angle(jnp.pi / 6, "rad"),
            "phi": u.Angle(jnp.array(0), "rad"),
        }

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

        assert isinstance(result, QMatrix)
        assert result.shape == (2,)
        assert jnp.allclose(result.value, expected, atol=1e-6)
        assert result.unit[0] == u.unit("m2 / rad2")
        assert result.unit[1] == u.unit("m2 / rad2")
