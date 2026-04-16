"""Tests for the scale_factors() manifold API and wrappers."""

import jax
import jax.numpy as jnp

import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm
from coordinax.internal import QuantityMatrix


class TestScaleFactorsEuclidean:
    """Tests for scale_factors on Euclidean metrics and manifolds."""

    def test_cartesian_metric_returns_1d_quantitymatrix(self):
        metric = cxm.EuclideanMetric(3)
        at = {
            "x": u.Q(jnp.array(1.0), "m"),
            "y": u.Q(jnp.array(2.0), "m"),
            "z": u.Q(jnp.array(3.0), "m"),
        }

        result = cxm.scale_factors(metric, cxc.cart3d, at=at)

        assert isinstance(result, QuantityMatrix)
        assert result.shape == (3,)
        assert result.ndim == 1
        assert jnp.allclose(result.value, jnp.array([1.0, 1.0, 1.0]))
        assert all(result.unit[i] == u.unit("") for i in range(3))

    def test_spherical_metric_returns_metric_diagonal_entries(self):
        metric = cxm.EuclideanMetric(3)
        at = {
            "r": u.Q(jnp.array(2.0), "m"),
            "theta": u.Angle(jnp.pi / 6, "rad"),
            "phi": u.Angle(jnp.array(0.4), "rad"),
        }

        result = cxm.scale_factors(metric, cxc.sph3d, at=at)

        assert isinstance(result, QuantityMatrix)
        assert result.shape == (3,)
        assert jnp.allclose(result.value, jnp.array([1.0, 4.0, 1.0]), atol=1e-6)
        assert result.unit[0] == u.unit("")
        assert result.unit[1] == u.unit("m2 / rad2")
        assert result.unit[2] == u.unit("m2 / rad2")

    def test_manifold_wrapper_matches_metric(self):
        manifold = cxm.EuclideanManifold(3)
        at = {
            "r": u.Q(jnp.array(3.0), "km"),
            "theta": u.Angle(jnp.pi / 2, "rad"),
            "phi": u.Angle(jnp.array(0.0), "rad"),
        }

        result_metric = cxm.scale_factors(manifold.metric, cxc.sph3d, at=at)
        result_manifold = cxm.scale_factors(manifold, cxc.sph3d, at=at)

        assert isinstance(result_manifold, QuantityMatrix)
        assert jnp.allclose(result_manifold.value, result_metric.value)
        assert result_manifold.unit.to_string() == result_metric.unit.to_string()


class TestScaleFactorsGeneric:
    """Tests for generic metric-based scale_factors behavior."""

    def test_hyperspherical_bare_arrays_promote_to_quantitymatrix(self):
        metric = cxm.HyperSphericalMetric(ndim=2)
        at = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0.0)}

        result = cxm.scale_factors(metric, cxc.sph2, at=at)

        assert isinstance(result, QuantityMatrix)
        assert result.shape == (2,)
        assert jnp.allclose(result.value, jnp.array([1.0, 1.0]), atol=1e-6)
        assert all(result.unit[i] == u.unit("") for i in range(2))

    def test_generic_path_matches_metric_matrix_diag(self):
        metric = cxm.HyperSphericalMetric(ndim=2)
        at = {
            "theta": u.Angle(jnp.pi / 3, "rad"),
            "phi": u.Angle(jnp.array(0.1), "rad"),
        }

        expected_metric = metric.metric_matrix(cxc.sph2, at=at)
        assert isinstance(expected_metric, QuantityMatrix)
        expected = expected_metric.diag()
        result = cxm.scale_factors(metric, cxc.sph2, at=at)

        assert isinstance(result, QuantityMatrix)
        assert jnp.allclose(result.value, expected.value)
        assert result.unit.to_string() == expected.unit.to_string()

    def test_jit(self):
        metric = cxm.HyperSphericalMetric(ndim=2)

        @jax.jit
        def compute(at):
            return cxm.scale_factors(metric, cxc.sph2, at=at)

        at = {
            "theta": u.Angle(jnp.pi / 2, "rad"),
            "phi": u.Angle(jnp.array(0.0), "rad"),
        }
        result = compute(at)

        assert isinstance(result, QuantityMatrix)
        assert jnp.allclose(result.value, jnp.array([1.0, 1.0]), atol=1e-6)

    def test_vmap_values(self):
        metric = cxm.HyperSphericalMetric(ndim=2)
        thetas = jnp.array([jnp.pi / 6, jnp.pi / 4, jnp.pi / 2])

        def compute(theta):
            return cxm.scale_factors(
                metric,
                cxc.sph2,
                at={"theta": theta, "phi": jnp.array(0.0)},
            ).value

        result = jax.vmap(compute)(thetas)
        expected = jnp.stack([jnp.ones_like(thetas), jnp.sin(thetas) ** 2], axis=-1)
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_embedded_manifold_requires_induced_metric(self):
        manifold = cxm.EmbeddedManifold(
            intrinsic=cxm.HyperSphericalManifold(),
            ambient=cxm.EuclideanManifold(3),
            embed_map=cxm.TwoSphereIn3D(radius=u.Q(jnp.array(2.0), "m")),
        )
        assert isinstance(manifold.metric, cxm.InducedMetric)

        at = {
            "theta": u.Angle(jnp.pi / 6, "rad"),
            "phi": u.Angle(jnp.array(0.0), "rad"),
        }

        result = cxm.scale_factors(manifold, cxc.sph2, at=at)

        # A 2-sphere of radius R embedded in Euclidean 3-space has induced metric
        # diag(R^2, R^2 sin^2(theta)) in the (theta, phi) chart. Here R = 2 m,
        # so the first diagonal entry is always 4 m^2 / rad^2.
        #
        # At theta = pi/6, sin^2(theta) = 1/4, so the second diagonal entry is
        # 4 * 1/4 = 1 with the same units. Using a non-equatorial point makes it
        # clear that we are testing the induced metric of the embedded sphere,
        # not just a coincidental [4, 4] value at the equator.
        expected = jnp.array([4.0, 1.0])

        assert isinstance(result, QuantityMatrix)
        assert result.shape == (2,)
        assert jnp.allclose(result.value, expected, atol=1e-6)
        assert result.unit[0] == u.unit("m2 / rad2")
        assert result.unit[1] == u.unit("m2 / rad2")
