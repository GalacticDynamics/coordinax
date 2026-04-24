"""Tests for coordinax.metrics — AbstractMetric and concrete implementations.

All tests in this file are RED until the metrics module is implemented.
"""

import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm

# =============================================================================
# AbstractMetric contract
# =============================================================================


class TestAbstractMetricContract:
    """Every AbstractMetric subclass must satisfy these invariants."""

    @pytest.fixture(
        params=[
            "euclidean3d",
            "euclidean2d",
            "euclidean1d",
            "hyperspherical2d",
            "minkowski4d",
            "product3d",
        ],
    )
    def metric(self, request):
        metrics = {
            "euclidean3d": cxm.EuclideanMetric(3),
            "euclidean2d": cxm.EuclideanMetric(2),
            "euclidean1d": cxm.EuclideanMetric(1),
            "hyperspherical2d": cxm.HyperSphericalMetric(ndim=2),
            "minkowski4d": cxm.MinkowskiMetric(),
            "product3d": cxm.CartesianProductMetric(
                factors=(cxm.HyperSphericalMetric(2), cxm.EuclideanMetric(1))
            ),
        }
        return metrics[request.param]

    def test_has_ndim(self, metric):
        assert isinstance(metric.ndim, int)
        assert metric.ndim > 0

    def test_is_static(self, metric):
        """Metrics must be JAX-static (hashable, eq works)."""
        leaves, _ = jax.tree.flatten(metric)
        assert leaves == []  # static: no leaves

    def test_signature_is_tuple(self, metric):
        assert isinstance(metric.signature, tuple)

    def test_signature_length_matches_ndim(self, metric):
        assert len(metric.signature) == metric.ndim

    def test_signature_entries_are_pm_one(self, metric):
        assert all(s in (-1, 1) for s in metric.signature)


# =============================================================================
# AbstractDiagonalMetric contract
# =============================================================================


class TestAbstractDiagonalMetricContract:
    """Every AbstractDiagonalMetric subclass must report is_diagonal=True.

    This class tests the structural promise added by AbstractDiagonalMetric beyond
    what AbstractMetric already guarantees: is_diagonal() must return True for all
    valid (metric, chart, base-point) combinations, regardless of position or usys.
    """

    @pytest.fixture(
        params=[
            "euclidean3d_cart",
            "euclidean3d_sph",
            "euclidean2d_cart",
            "hyperspherical2d",
            "minkowski4d",
        ]
    )
    def metric_chart_at(self, request):
        cases = {
            "euclidean3d_cart": (
                cxm.EuclideanMetric(3),
                cxc.cart3d,
                {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")},
            ),
            "euclidean3d_sph": (
                cxm.EuclideanMetric(3),
                cxc.sph3d,
                {
                    "r": u.Q(2.0, "m"),
                    "theta": u.Angle(jnp.pi / 3, "rad"),
                    "phi": u.Angle(1.0, "rad"),
                },
            ),
            "euclidean2d_cart": (
                cxm.EuclideanMetric(2),
                cxc.cart2d,
                {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")},
            ),
            "hyperspherical2d": (
                cxm.HyperSphericalMetric(ndim=2),
                cxc.sph2,
                {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")},
            ),
            "minkowski4d": (
                cxm.MinkowskiMetric(),
                cxc.SpaceTimeCT(cxc.cart3d),
                {
                    "ct": u.Q(1.0, "m"),
                    "x": u.Q(0.0, "m"),
                    "y": u.Q(0.0, "m"),
                    "z": u.Q(0.0, "m"),
                },
            ),
        }
        return cases[request.param]

    def test_is_instance_of_abstractdiagonalmetric(self, metric_chart_at):
        metric, _, _ = metric_chart_at
        assert isinstance(metric, cxm.AbstractDiagonalMetric)

    def test_is_diagonal_returns_true(self, metric_chart_at):
        metric, chart, at = metric_chart_at
        result = metric.is_diagonal(chart, at=at)
        assert bool(result) is True

    def test_is_diagonal_output_shape(self, metric_chart_at):
        metric, chart, at = metric_chart_at
        result = metric.is_diagonal(chart, at=at)
        assert result.shape == ()
        assert result.dtype == jnp.bool_

    def test_is_diagonal_under_jit(self, metric_chart_at):
        metric, chart, at = metric_chart_at

        @jax.jit
        def compute(at):
            return metric.is_diagonal(chart, at=at)

        result = compute(at)
        assert bool(result) is True

    def test_is_diagonal_ignores_usys(self, metric_chart_at):
        """is_diagonal must return True regardless of usys argument."""
        metric, chart, at = metric_chart_at
        result_no_usys = metric.is_diagonal(chart, at=at)
        result_with_usys = metric.is_diagonal(chart, at=at, usys=u.unitsystems.si)
        assert bool(result_no_usys) is True
        assert bool(result_with_usys) is True


# =============================================================================
# EuclideanMetric
# =============================================================================


class TestEuclideanMetric:
    """Tests for EuclideanMetric, the flat Riemannian metric on Euclidean space."""

    def test_isinstance_abstractdiagonalmetric(self):
        assert isinstance(cxm.EuclideanMetric(3), cxm.AbstractDiagonalMetric)

    def test_construction_1d(self):
        m = cxm.EuclideanMetric(1)
        assert m.ndim == 1

    def test_construction_2d(self):
        m = cxm.EuclideanMetric(2)
        assert m.ndim == 2

    def test_construction_3d(self):
        m = cxm.EuclideanMetric(3)
        assert m.ndim == 3

    def test_metric_matrix_cart3d_is_identity(self):
        m = cxm.EuclideanMetric(3)
        p = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}
        g = m.metric_matrix(cxc.cart3d, at=p)
        assert g.shape == (3, 3)
        assert jnp.allclose(g.value, jnp.eye(3))

    def test_metric_matrix_cart2d_is_identity(self):
        m = cxm.EuclideanMetric(2)
        p = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
        g = m.metric_matrix(cxc.cart2d, at=p)
        assert g.shape == (2, 2)
        assert jnp.allclose(g.value, jnp.eye(2))

    def test_metric_matrix_sph3d_at_origin(self):
        """Spherical metric at (1, pi/2, 0): diag(1, r^2, r^2 sin^2 theta)."""
        m = cxm.EuclideanMetric(3)
        p = {
            "r": u.Q(1.0, "m"),
            "theta": u.Angle(jnp.pi / 2, "rad"),
            "phi": u.Angle(0.0, "rad"),
        }
        g = m.metric_matrix(cxc.sph3d, at=p)
        assert g.shape == (3, 3)
        # diagonal entries: g_rr=1, g_tt=r^2=1, g_pp=r^2 sin^2(theta)=1
        expected_diag = jnp.array([1.0, 1.0, 1.0])
        assert jnp.allclose(jnp.diag(g.value), expected_diag, atol=1e-6)

    def test_metric_matrix_sph3d_diagonal(self):
        """Spherical metric is always diagonal."""
        m = cxm.EuclideanMetric(3)
        p = {
            "r": u.Q(2.0, "m"),
            "theta": u.Angle(jnp.pi / 3, "rad"),
            "phi": u.Angle(1.0, "rad"),
        }
        g = m.metric_matrix(cxc.sph3d, at=p)
        # Off-diagonal elements must be zero
        offdiag = g.value - jnp.diag(jnp.diag(g.value))
        assert jnp.allclose(offdiag, jnp.zeros((3, 3)), atol=1e-6)

    def test_metric_matrix_jit(self):
        m = cxm.EuclideanMetric(3)
        p = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}

        @jax.jit
        def compute(p):
            return m.metric_matrix(cxc.cart3d, at=p)

        g = compute(p)
        assert g.shape == (3, 3)

    def test_carried_by_euclidean_manifold(self):
        """euclidean3d.metric should return an EuclideanMetric."""
        metric = cxm.euclidean3d.metric
        assert isinstance(metric, cxm.EuclideanMetric)
        assert metric.ndim == 3


# =============================================================================
# HyperSphericalMetric
# =============================================================================


class TestHyperSphericalMetric:
    """Tests for HyperSphericalMetric, the round metric on the unit sphere."""

    def test_isinstance_abstractdiagonalmetric(self):
        assert isinstance(cxm.HyperSphericalMetric(ndim=2), cxm.AbstractDiagonalMetric)

    def test_construction(self):
        m = cxm.HyperSphericalMetric(ndim=2)
        assert m.ndim == 2

    def test_metric_matrix_at_equator(self):
        """S^2 metric at equator: diag(1, sin^2(theta)) = diag(1, 1)."""
        m = cxm.HyperSphericalMetric(ndim=2)
        p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
        g = m.metric_matrix(cxc.sph2, at=p)
        assert g.shape == (2, 2)
        expected = jnp.diag(jnp.array([1.0, 1.0]))
        assert jnp.allclose(g.value, expected, atol=1e-6)

    def test_metric_matrix_at_pole_theta_component(self):
        """S^2 metric g_theta_theta = 1 everywhere."""
        m = cxm.HyperSphericalMetric(ndim=2)
        p = {"theta": u.Angle(0.1, "rad"), "phi": u.Angle(0.0, "rad")}
        g = m.metric_matrix(cxc.sph2, at=p)
        assert jnp.allclose(g.value[0, 0], 1.0, atol=1e-6)

    def test_metric_matrix_phi_component_at_various_latitudes(self):
        """g_phi_phi = sin^2(theta)."""
        m = cxm.HyperSphericalMetric(ndim=2)
        for theta_val in [0.1, jnp.pi / 4, jnp.pi / 2, jnp.pi * 3 / 4]:
            p = {"theta": u.Angle(theta_val, "rad"), "phi": u.Angle(0.0, "rad")}
            g = m.metric_matrix(cxc.sph2, at=p)
            expected_g11 = jnp.sin(theta_val) ** 2
            assert jnp.allclose(g.value[1, 1], expected_g11, atol=1e-6), (
                f"theta={theta_val}: g_phi_phi={g[1, 1]} != sin^2(theta)={expected_g11}"
            )

    def test_metric_matrix_is_diagonal(self):
        """S^2 metric matrix is always diagonal."""
        m = cxm.HyperSphericalMetric(ndim=2)
        p = {"theta": u.Angle(jnp.pi / 3, "rad"), "phi": u.Angle(1.0, "rad")}
        g = m.metric_matrix(cxc.sph2, at=p)
        offdiag = g.value - jnp.diag(jnp.diag(g.value))
        assert jnp.allclose(offdiag, jnp.zeros((2, 2)), atol=1e-6)

    def test_metric_matrix_jit(self):
        m = cxm.HyperSphericalMetric(ndim=2)
        p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}

        @jax.jit
        def compute(p):
            return m.metric_matrix(cxc.sph2, at=p)

        g = compute(p)
        assert g.shape == (2, 2)

    def test_metric_matrix_vmap(self):
        m = cxm.HyperSphericalMetric(ndim=2)
        thetas = jnp.linspace(0.1, jnp.pi - 0.1, 5)

        def single(theta_val):
            p = {"theta": u.Angle(theta_val, "rad"), "phi": u.Angle(0.0, "rad")}
            return m.metric_matrix(cxc.sph2, at=p)

        gs = jax.vmap(single)(thetas)
        assert gs.shape == (5, 2, 2)

    def test_carried_by_hyperspherical_manifold(self):
        """twosphere.metric should return a HyperSphericalMetric."""
        metric = cxm.twosphere.metric
        assert isinstance(metric, cxm.HyperSphericalMetric)
        assert metric.ndim == 2


# =============================================================================
# MinkowskiMetric
# =============================================================================


class TestMinkowskiMetric:
    """Tests for MinkowskiMetric, the flat Lorentzian metric on Minkowski spacetime."""

    def test_isinstance_abstractdiagonalmetric(self):
        assert isinstance(cxm.MinkowskiMetric(), cxm.AbstractDiagonalMetric)

    def test_construction(self):
        M = cxm.MinkowskiMetric()
        assert M.ndim == 4

    def test_metric_matrix_is_diagonal(self):
        m = cxm.MinkowskiMetric()
        chart = cxc.SpaceTimeCT(cxc.cart3d)
        p = {
            "ct": u.Q(1.0, "m"),
            "x": u.Q(0.0, "m"),
            "y": u.Q(0.0, "m"),
            "z": u.Q(0.0, "m"),
        }
        g = m.metric_matrix(chart, at=p)
        assert g.shape == (4, 4)
        expected = jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0]))
        assert jnp.allclose(g.value, expected, atol=1e-6)

    def test_metric_matrix_is_position_independent(self):
        """Minkowski metric is flat — does not depend on position."""
        m = cxm.MinkowskiMetric()
        chart = cxc.SpaceTimeCT(cxc.cart3d)
        p1 = {
            "ct": u.Q(1.0, "m"),
            "x": u.Q(0.0, "m"),
            "y": u.Q(0.0, "m"),
            "z": u.Q(0.0, "m"),
        }
        p2 = {
            "ct": u.Q(100.0, "m"),
            "x": u.Q(50.0, "m"),
            "y": u.Q(-30.0, "m"),
            "z": u.Q(20.0, "m"),
        }
        g1 = m.metric_matrix(chart, at=p1)
        g2 = m.metric_matrix(chart, at=p2)
        assert jnp.allclose(g1.value, g2.value, atol=1e-6)

    def test_metric_matrix_jit(self):
        m = cxm.MinkowskiMetric()
        chart = cxc.SpaceTimeCT(cxc.cart3d)
        p = {
            "ct": u.Q(1.0, "m"),
            "x": u.Q(0.0, "m"),
            "y": u.Q(0, "m"),
            "z": u.Q(0, "m"),
        }

        @jax.jit
        def compute(p):
            return m.metric_matrix(chart, at=p)

        g = compute(p)
        assert g.shape == (4, 4)

    def test_carried_by_minkowski_manifold(self):
        """minkowski4d.metric should return a MinkowskiMetric."""
        metric = cxm.minkowski4d.metric
        assert isinstance(metric, cxm.MinkowskiMetric)
        assert metric.ndim == 4


# =============================================================================
# InducedMetric
# =============================================================================


class TestInducedMetric:
    """Tests for InducedMetric, the pullback metric on an embedded manifold."""

    def test_induced_metric_is_not_abstractdiagonalmetric(self):
        """InducedMetric is NOT an AbstractDiagonalMetric."""
        manifold = cxm.embedded_twosphere(radius=1.0)
        assert not isinstance(manifold.metric, cxm.AbstractDiagonalMetric)

    def test_unit_sphere_at_equator(self):
        """Induced metric on S^2 embedded in R^3 at equator matches sphere metric."""
        manifold = cxm.embedded_twosphere(radius=1.0)
        metric = manifold.metric
        assert isinstance(metric, cxm.InducedMetric)

        p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
        g = metric.metric_matrix(cxc.sph2, at=p)
        assert g.shape == (2, 2)
        # At equator: diag(1, sin^2(pi/2)) = diag(1, 1)
        expected = jnp.eye(2)
        assert jnp.allclose(g.value, expected, atol=1e-4)

    def test_radius_2_sphere_at_equator(self):
        """Induced metric on radius-2 sphere.

        diag(R^2, R^2 sin^2 theta) at equator = diag(4, 4).
        """
        manifold = cxm.EmbeddedManifold(
            intrinsic=cxm.HyperSphericalManifold(),
            ambient=cxm.EuclideanManifold(3),
            embed_map=cxm.TwoSphereIn3D(radius=u.Q(2.0, "m")),
        )
        metric = manifold.metric
        assert isinstance(metric, cxm.InducedMetric)

        p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
        g = metric.metric_matrix(cxc.sph2, at=p)
        assert g.shape == (2, 2)
        expected = jnp.diag(jnp.array([4.0, 4.0]))
        assert jnp.allclose(g.value, expected, atol=1e-3)

    def test_induced_metric_jit(self):
        """InducedMetric.metric_matrix should work under jit."""
        manifold = cxm.embedded_twosphere(radius=1.0)
        metric = manifold.metric

        @jax.jit
        def compute(p):
            return metric.metric_matrix(cxc.sph2, at=p)

        p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
        g = compute(p)
        assert g.shape == (2, 2)


# =============================================================================
# CartesianProductMetric
# =============================================================================


class TestCartesianProductMetric:
    """Tests for CartesianProductMetric on product manifolds."""

    def test_product_metric_is_not_abstractdiagonalmetric(self):
        """CartesianProductMetric is NOT an AbstractDiagonalMetric."""
        M = cxm.CartesianProductManifold(
            factors=(cxm.HyperSphericalManifold(), cxm.EuclideanManifold(1)),
            factor_names=("S2", "R1"),
        )
        assert not isinstance(M.metric, cxm.AbstractDiagonalMetric)

    def test_signature_concatenates_factor_signatures(self):
        M = cxm.CartesianProductManifold(
            factors=(cxm.MinkowskiManifold(), cxm.EuclideanManifold(1)),
            factor_names=("st", "x"),
        )
        metric = M.metric
        assert isinstance(metric, cxm.CartesianProductMetric)
        assert metric.signature == (-1, 1, 1, 1, 1)
        assert metric.ndim == 5

    def test_metric_matrix_is_block_diagonal_in_product_chart(self):
        M = cxm.CartesianProductManifold(
            factors=(cxm.HyperSphericalManifold(), cxm.EuclideanManifold(1)),
            factor_names=("S2", "R1"),
        )
        chart = cxc.CartesianProductChart((cxc.sph2, cxc.cart1d), ("S2", "R1"))
        p = {
            "S2.theta": u.Angle(jnp.pi / 2, "rad"),
            "S2.phi": u.Angle(0.0, "rad"),
            "R1.x": u.Q(1.0, "m"),
        }

        g = M.metric.metric_matrix(chart, at=p)
        assert g.shape == (3, 3)
        assert jnp.allclose(g.value, jnp.eye(3), atol=1e-6)

    def test_product_manifold_carries_product_metric(self):
        M = cxm.CartesianProductManifold(
            factors=(cxm.HyperSphericalManifold(), cxm.EuclideanManifold(1)),
            factor_names=("S2", "R1"),
        )
        assert isinstance(M.metric, cxm.CartesianProductMetric)
