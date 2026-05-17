"""Pullback metric consistency: RoundMetric vs Jacobian pullback on S².

For the unit two-sphere S², the round metric in (θ, φ) coordinates gives
g = diag(1, sin²θ).  The same result must follow from the Jacobian pullback
of the flat metric on R³ via the standard Cartesian embedding:

    (θ, φ) → (sin(θ)cos(φ), sin(θ)sin(φ), cos(θ))

These tests assert that both paths agree numerically at sample points and
across a range of angles verified with Hypothesis.
"""

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings

import unxt as u

import coordinax.api.manifolds as cxmapi
import coordinax.charts as cxc
import coordinax.manifolds as cxm
from coordinax._src.metric.matrix import DenseMetric, DiagonalMetric

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def unit_sphere_embedded():
    """EmbeddedManifold for the unit two-sphere (dimensionless, radius=1)."""
    return cxm.EmbeddedManifold(
        intrinsic=cxm.S2,
        ambient=cxm.R3,
        embed_map=cxm.TwoSphereIn3D(radius=1.0),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _round_dense_matrix(theta, phi):
    """Expected round-metric as a dense 2×2 array at (theta, phi)."""
    return jnp.array([[1.0, 0.0], [0.0, jnp.sin(theta) ** 2]])


# ---------------------------------------------------------------------------
# Type contract tests
# ---------------------------------------------------------------------------


class TestPullbackConsistencyTypes:
    """Verify the metric_matrix return types for both paths."""

    def test_round_metric_returns_diagonal(self):
        pt = {"theta": jnp.array(jnp.pi / 3), "phi": jnp.array(jnp.pi / 4)}
        g = cxmapi.metric_matrix(cxm.S2, pt, cxc.sph2)
        assert isinstance(g, DiagonalMetric)

    def test_embedded_metric_returns_dense(self, unit_sphere_embedded):
        pt = {"theta": jnp.array(jnp.pi / 3), "phi": jnp.array(jnp.pi / 4)}
        g = cxmapi.metric_matrix(unit_sphere_embedded, pt, cxc.sph2)
        assert isinstance(g, DenseMetric)


# ---------------------------------------------------------------------------
# Numerical consistency tests
# ---------------------------------------------------------------------------


class TestPullbackConsistencyNumerical:
    """Both paths give the same metric matrix at sample points."""

    @pytest.mark.parametrize(
        ("theta", "phi"),
        [
            (jnp.pi / 2, 0.0),  # equator, phi=0
            (jnp.pi / 3, jnp.pi / 4),  # off-equator
            (jnp.pi / 6, jnp.pi),  # high latitude, phi=π
            (0.1, 2.5),  # near pole, arbitrary phi
        ],
        ids=["equator-0", "off-equator", "high-lat-pi", "near-pole"],
    )
    def test_sample_point(self, unit_sphere_embedded, theta, phi):
        pt = {"theta": jnp.array(theta), "phi": jnp.array(phi)}

        g_round = cxmapi.metric_matrix(cxm.S2, pt, cxc.sph2)
        g_pullback = cxmapi.metric_matrix(unit_sphere_embedded, pt, cxc.sph2)

        # RoundMetric (diagonal) and Jacobian pullback (dense) must agree.
        expected = g_round.to_dense().matrix  # plain array, shape (2, 2)
        actual = g_pullback.matrix.value  # QMatrix.value, shape (2, 2)

        assert jnp.allclose(actual, expected, atol=1e-6), (
            f"Mismatch at theta={theta}, phi={phi}:\n"
            f"  expected={expected}\n  actual={actual}"
        )

    @given(
        theta=st.floats(
            min_value=0.05, max_value=3.09, allow_nan=False, allow_infinity=False
        ),
        phi=st.floats(
            min_value=0.0, max_value=6.28, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_hypothesis_s2(self, unit_sphere_embedded, theta, phi):
        pt = {"theta": jnp.array(theta), "phi": jnp.array(phi)}

        g_round = cxmapi.metric_matrix(cxm.S2, pt, cxc.sph2)
        g_pullback = cxmapi.metric_matrix(unit_sphere_embedded, pt, cxc.sph2)

        expected = g_round.to_dense().matrix
        actual = g_pullback.matrix.value

        assert jnp.allclose(actual, expected, atol=1e-5), (
            f"Mismatch at theta={theta:.4f}, phi={phi:.4f}:\n"
            f"  expected={expected}\n  actual={actual}"
        )


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


class TestPullbackConsistencyJIT:
    """Both metric paths are JIT-compatible."""

    def test_round_metric_jit(self):
        @jax.jit
        def compute(theta, phi):
            pt = {"theta": theta, "phi": phi}
            return cxmapi.metric_matrix(cxm.S2, pt, cxc.sph2).diagonal

        result = compute(jnp.array(jnp.pi / 3), jnp.array(0.0))
        assert result.shape == (2,)

    def test_pullback_metric_jit(self):
        M = cxm.EmbeddedManifold(
            intrinsic=cxm.S2,
            ambient=cxm.R3,
            embed_map=cxm.TwoSphereIn3D(radius=1.0),
        )

        @jax.jit
        def compute(theta, phi):
            pt = {"theta": theta, "phi": phi}
            return cxmapi.metric_matrix(M, pt, cxc.sph2).matrix.value

        result = compute(jnp.array(jnp.pi / 3), jnp.array(0.0))
        assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# Unit preservation for non-trivial radius
# ---------------------------------------------------------------------------


class TestPullbackMetricUnits:
    """For a sphere with physical radius, the metric carries correct units."""

    def test_radius_1km_at_equator(self):
        M = cxm.EmbeddedManifold(
            intrinsic=cxm.S2,
            ambient=cxm.R3,
            embed_map=cxm.TwoSphereIn3D(radius=u.Q(1.0, "km")),
        )
        at = {"theta": u.Q(jnp.pi / 2, "rad"), "phi": u.Q(0.0, "rad")}
        g = cxmapi.metric_matrix(M, at, cxc.sph2)
        assert isinstance(g, DenseMetric)
        # At the equator sin(π/2)=1, so metric should be identity × km²/rad²
        assert jnp.allclose(g.matrix.value, jnp.eye(2), atol=1e-6)
        assert str(g.matrix.unit[0, 0]) == "km2 / rad2"

    def test_radius_2m_metric_scaled(self):
        M = cxm.EmbeddedManifold(
            intrinsic=cxm.S2,
            ambient=cxm.R3,
            embed_map=cxm.TwoSphereIn3D(radius=u.Q(2.0, "m")),
        )
        at = {"theta": u.Q(jnp.pi / 2, "rad"), "phi": u.Q(0.0, "rad")}
        g = cxmapi.metric_matrix(M, at, cxc.sph2)
        # Metric = R² × I at equator → values should be [[4, 0], [0, 4]]
        assert jnp.allclose(g.matrix.value, 4.0 * jnp.eye(2), atol=1e-6)
        assert str(g.matrix.unit[0, 0]) == "m2 / rad2"
