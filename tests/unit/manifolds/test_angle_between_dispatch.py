"""Tests for the angle_between() manifold API and wrappers."""

import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm
from coordinax.angles import AbstractAngle


class TestAngleBetweenEuclidean:
    """Tests for angle_between on Euclidean metrics and manifolds."""

    def test_cartesian_right_angle_returns_angle(self):
        metric = cxm.EuclideanMetric(2)
        at = {"x": u.Q(jnp.array(0.0), "m"), "y": u.Q(jnp.array(0.0), "m")}
        uvec = {"x": u.Q(jnp.array(1.0), "m"), "y": u.Q(jnp.array(0.0), "m")}
        vvec = {"x": u.Q(jnp.array(0.0), "m"), "y": u.Q(jnp.array(2.0), "m")}

        got = cxm.angle_between(metric, cxc.cart2d, uvec, vvec, at=at)

        assert isinstance(got, AbstractAngle)
        assert jnp.allclose(u.ustrip("rad", got), jnp.pi / 2, atol=1e-6)


class TestAngleBetweenFailureModes:
    """Tests for invalid inputs and unsupported metrics."""

    def test_zero_norm_vector_raises_value_error(self):
        metric = cxm.EuclideanMetric(2)
        at = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        zero = {"x": jnp.array(0.0), "y": jnp.array(0.0)}
        other = {"x": jnp.array(1.0), "y": jnp.array(0.0)}

        with pytest.raises(ValueError, match="zero"):
            cxm.angle_between(metric, cxc.cart2d, zero, other, at=at)

    def test_indefinite_metric_is_not_supported(self):
        metric = cxm.MinkowskiMetric()
        at = {
            "ct": jnp.array(0.0),
            "x": jnp.array(0.0),
            "y": jnp.array(0.0),
            "z": jnp.array(0.0),
        }
        uvec = {
            "ct": jnp.array(0.0),
            "x": jnp.array(1.0),
            "y": jnp.array(0.0),
            "z": jnp.array(0.0),
        }
        vvec = {
            "ct": jnp.array(0.0),
            "x": jnp.array(0.0),
            "y": jnp.array(1.0),
            "z": jnp.array(0.0),
        }

        with pytest.raises(NotImplementedError, match=r"pseudo.*indefinite"):
            cxm.angle_between(metric, cxc.minkowskict, uvec, vvec, at=at)


class TestAngleBetweenJAX:
    """Tests for JAX compatibility of angle_between."""

    def test_jit(self):
        metric = cxm.HyperSphericalMetric(ndim=2)

        @jax.jit
        def compute(theta):
            at = {"theta": theta, "phi": jnp.array(0.0)}
            uvec = {"theta": jnp.array(1.0), "phi": jnp.array(0.0)}
            vvec = {"theta": jnp.array(1.0), "phi": jnp.array(1.0)}
            return u.ustrip(
                "rad", cxm.angle_between(metric, cxc.sph2, uvec, vvec, at=at)
            )

        got = compute(jnp.array(jnp.pi / 2))
        assert jnp.allclose(got, jnp.pi / 4, atol=1e-6)

    def test_vmap_values(self):
        metric = cxm.HyperSphericalMetric(ndim=2)
        thetas = jnp.array([jnp.pi / 6, jnp.pi / 4, jnp.pi / 2])

        def compute(theta):
            at = {"theta": theta, "phi": jnp.array(0.0)}
            uvec = {"theta": jnp.array(1.0), "phi": jnp.array(0.0)}
            vvec = {"theta": jnp.array(1.0), "phi": jnp.array(1.0)}
            return u.ustrip(
                "rad", cxm.angle_between(metric, cxc.sph2, uvec, vvec, at=at)
            )

        got = jax.vmap(compute)(thetas)
        expected = jnp.arccos(1.0 / jnp.sqrt(1.0 + jnp.sin(thetas) ** 2))
        assert jnp.allclose(got, expected, atol=1e-6)
