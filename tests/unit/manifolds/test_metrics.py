"""Tests for coordinax.metrics — AbstractMetric and concrete implementations.

All tests in this file are RED until the metrics module is implemented.
"""

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, settings

import unxt as u

import coordinax.charts as cxc
import coordinax.internal as cxi
import coordinax.manifolds as cxm

# =============================================================================
# AbstractMetric contract
# =============================================================================


class TestAbstractMetricContract:
    """Every AbstractMetric subclass must satisfy these invariants."""

    @pytest.fixture(
        params=[
            "R3",
            "R2",
            "R1",
            "hyperspherical2d",
            "minkowski4d",
            "product3d",
        ],
    )
    def metric(self, request):
        metrics = {
            "R3": cxm.EuclideanMetric(3),
            "R2": cxm.EuclideanMetric(2),
            "R1": cxm.EuclideanMetric(1),
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
            "R3_cart",
            "R3_sph",
            "R2_cart",
            "hyperspherical2d",
            "minkowski4d",
        ]
    )
    def metric_chart_at(self, request):
        cases = {
            "R3_cart": (
                cxm.EuclideanMetric(3),
                cxc.cart3d,
                {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")},
            ),
            "R3_sph": (
                cxm.EuclideanMetric(3),
                cxc.sph3d,
                {
                    "r": u.Q(2.0, "m"),
                    "theta": u.Angle(jnp.pi / 3, "rad"),
                    "phi": u.Angle(1.0, "rad"),
                },
            ),
            "R2_cart": (
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
        """R3.metric should return an EuclideanMetric."""
        metric = cxm.R3.metric
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
        """S2.metric should return a HyperSphericalMetric."""
        metric = cxm.S2.metric
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


# =============================================================================
# AbstractMetric.cholesky
# =============================================================================


class TestAbstractMetricCholesky:
    """Tests for AbstractMetric.cholesky — Cholesky factorization L of the metric.

    The factorization satisfies g = L L^T where L is lower-triangular with
    strictly positive diagonal entries.  Tests are restricted to
    positive-definite (Riemannian) metrics; indefinite metrics (Minkowski) are
    excluded because jnp.linalg.cholesky requires positive-definiteness.
    """

    @pytest.fixture(
        params=[
            "R3_cart",
            "R3_sph",
            "R2_cart",
            "hyperspherical2d",
        ]
    )
    def metric_chart_at(self, request):
        cases = {
            "R3_cart": (
                cxm.EuclideanMetric(3),
                cxc.cart3d,
                {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")},
            ),
            "R3_sph": (
                cxm.EuclideanMetric(3),
                cxc.sph3d,
                {
                    "r": u.Q(2.0, "m"),
                    "theta": u.Angle(jnp.pi / 3, "rad"),
                    "phi": u.Angle(1.0, "rad"),
                },
            ),
            "R2_cart": (
                cxm.EuclideanMetric(2),
                cxc.cart2d,
                {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")},
            ),
            "hyperspherical2d": (
                cxm.HyperSphericalMetric(ndim=2),
                cxc.sph2,
                {"theta": u.Angle(jnp.pi / 3, "rad"), "phi": u.Angle(0.5, "rad")},
            ),
        }
        return cases[request.param]

    def test_returns_quantitymatrix(self, metric_chart_at):
        """cholesky() returns a QuantityMatrix for unitful metrics."""
        metric, chart, at = metric_chart_at
        L = metric.cholesky(chart, at=at)
        assert isinstance(L, cxi.QuantityMatrix)

    def test_shape(self, metric_chart_at):
        """cholesky() result has shape (n, n)."""
        metric, chart, at = metric_chart_at
        L = metric.cholesky(chart, at=at)
        n = metric.ndim
        assert L.shape == (n, n)

    def test_lower_triangular(self, metric_chart_at):
        """All entries strictly above the diagonal must be zero."""
        metric, chart, at = metric_chart_at
        L = metric.cholesky(chart, at=at)
        upper = jnp.triu(L.value, k=1)
        assert jnp.allclose(upper, jnp.zeros_like(upper), atol=1e-6)

    def test_positive_diagonal(self, metric_chart_at):
        """All diagonal entries of L must be strictly positive."""
        metric, chart, at = metric_chart_at
        L = metric.cholesky(chart, at=at)
        assert jnp.all(jnp.diag(L.value) > 0)

    def test_reconstruction(self, metric_chart_at):
        """L @ L.T must equal the original metric matrix G (values and units)."""
        metric, chart, at = metric_chart_at
        L = metric.cholesky(chart, at=at)
        G = metric.metric_matrix(chart, at=at)
        assert isinstance(L, cxi.QuantityMatrix)
        # Verify units: each L[i,j] must carry sqrt(G[i,j]) units
        n = G.value.shape[0]
        for i in range(n):
            for j in range(n):
                assert L.unit[i][j] ** 2 == G.unit[i][j]
        # Verify numeric reconstruction: L @ L^T == G
        assert jnp.allclose(L.value @ L.value.T, G.value, atol=1e-6)

    def test_cartesian_euclidean_is_identity(self):
        """Cholesky of the identity metric (Cartesian Euclidean) is the identity."""
        metric = cxm.EuclideanMetric(3)
        at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
        L = metric.cholesky(cxc.cart3d, at=at)
        assert jnp.allclose(L.value, jnp.eye(3), atol=1e-6)

    def test_diagonal_metric_cholesky_is_diagonal(self, metric_chart_at):
        """For diagonal metrics, L is also diagonal (L_ii = sqrt(g_ii))."""
        metric, chart, at = metric_chart_at
        G = metric.metric_matrix(chart, at=at)
        if not metric.is_diagonal(chart, at=at):
            pytest.skip("metric is not diagonal at this chart/point")
        L = metric.cholesky(chart, at=at)
        # Off-diagonal entries of L must be zero
        offdiag = L.value - jnp.diag(jnp.diag(L.value))
        assert jnp.allclose(offdiag, jnp.zeros_like(offdiag), atol=1e-6)
        # Diagonal entries must equal sqrt(g_ii)
        expected_diag = jnp.sqrt(jnp.diag(G.value))
        assert jnp.allclose(jnp.diag(L.value), expected_diag, atol=1e-6)

    def test_jit(self, metric_chart_at):
        """cholesky() must be compatible with jax.jit."""
        metric, chart, at = metric_chart_at

        @jax.jit
        def compute(at):
            return metric.cholesky(chart, at=at)

        L = compute(at)
        assert L.shape == (metric.ndim, metric.ndim)


# =============================================================================
# AbstractMetric.is_diagonal  (base-class implementation)
# =============================================================================


class TestAbstractMetricIsDiagonal:
    """Tests for AbstractMetric.is_diagonal — the matrix-based base implementation.

    AbstractMetric.is_diagonal evaluates the metric matrix at a specific point
    and checks whether all off-diagonal entries satisfy ``jnp.allclose(..., 0)``
    (i.e. within default floating-point tolerance, NOT exact equality).

    AbstractDiagonalMetric overrides this to return True unconditionally without
    evaluating the matrix; that behaviour is covered by
    TestAbstractDiagonalMetricContract.

    Note: the InducedMetric on S^2 in the sph2 chart has off-diagonal entries at
    machine-epsilon level (~1e-17) everywhere, so is_diagonal always returns True
    for that metric.
    """

    # ------------------------------------------------------------------
    # True cases: orthogonal metrics
    # ------------------------------------------------------------------

    def test_induced_metric_on_sphere_is_diagonal(self):
        """InducedMetric on S^2 in sph2 is diagonal everywhere (off-diag ~1e-17)."""
        metric = cxm.embedded_twosphere(radius=1.0).metric
        assert not isinstance(metric, cxm.AbstractDiagonalMetric)
        p = {"theta": u.Angle(jnp.pi / 3, "rad"), "phi": u.Angle(1.0, "rad")}
        result = metric.is_diagonal(cxc.sph2, at=p)
        assert bool(result) is True

    def test_product_metric_euclidean_cart_is_diagonal(self):
        """Euclidean factors in Cartesian chart is diagonal."""
        M = cxm.CartesianProductManifold(
            factors=(cxm.EuclideanManifold(2), cxm.EuclideanManifold(1)),
            factor_names=("xy", "z"),
        )
        chart = cxc.CartesianProductChart((cxc.cart2d, cxc.cart1d), ("xy", "z"))
        p = {"xy.x": u.Q(1.0, "m"), "xy.y": u.Q(2.0, "m"), "z.x": u.Q(3.0, "m")}
        result = M.metric.is_diagonal(chart, at=p)
        assert bool(result) is True

    # ------------------------------------------------------------------
    # Output contract
    # ------------------------------------------------------------------

    def test_returns_scalar_bool_array(self):
        """is_diagonal() must return a scalar Array with bool dtype."""
        metric = cxm.embedded_twosphere(radius=1.0).metric
        p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
        result = metric.is_diagonal(cxc.sph2, at=p)
        assert result.shape == ()
        assert result.dtype == jnp.bool_

    def test_jit_compatible(self):
        """is_diagonal() must work under jax.jit."""
        metric = cxm.embedded_twosphere(radius=1.0).metric
        p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}

        @jax.jit
        def compute(p):
            return metric.is_diagonal(cxc.sph2, at=p)

        result = compute(p)
        assert result.shape == ()
        assert result.dtype == jnp.bool_

    # ------------------------------------------------------------------
    # Consistency: is_diagonal agrees with allclose off-diagonal check
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("metric", "chart", "at"),
        [
            # CartesianProductMetric of Euclidean spaces — exactly diagonal
            (
                cxm.CartesianProductManifold(
                    factors=(cxm.EuclideanManifold(2), cxm.EuclideanManifold(1)),
                    factor_names=("xy", "z"),
                ).metric,
                cxc.CartesianProductChart((cxc.cart2d, cxc.cart1d), ("xy", "z")),
                {"xy.x": u.Q(1.0, "m"), "xy.y": u.Q(2.0, "m"), "z.x": u.Q(3.0, "m")},
            ),
            # InducedMetric at phi=0 — exactly diagonal
            (
                cxm.embedded_twosphere(radius=1.0).metric,
                cxc.sph2,
                {"theta": u.Angle(jnp.pi / 4, "rad"), "phi": u.Angle(0.0, "rad")},
            ),
            # InducedMetric at phi=1.0 — off-diag ~1e-17 (allclose → True)
            (
                cxm.embedded_twosphere(radius=1.0).metric,
                cxc.sph2,
                {"theta": u.Angle(jnp.pi / 3, "rad"), "phi": u.Angle(1.0, "rad")},
            ),
            # EuclideanMetric (AbstractDiagonalMetric) in Cartesian
            (
                cxm.EuclideanMetric(3),
                cxc.cart3d,
                {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")},
            ),
        ],
        ids=["product_euclidean", "induced_phi0", "induced_phi1", "euclidean_cart"],
    )
    def test_consistent_with_allclose_matrix_check(self, metric, chart, at):
        """is_diagonal() must agree with jnp.allclose(offdiag, 0) on the matrix.

        The base implementation uses allclose (not exact equality).
        """
        result = metric.is_diagonal(chart, at=at)
        G = metric.metric_matrix(chart, at=at)
        val = G.value if hasattr(G, "value") else G
        offdiag = val - jnp.diag(jnp.diag(val))
        expected = jnp.allclose(offdiag, jnp.zeros_like(offdiag))
        assert bool(result) == bool(expected)

    # ------------------------------------------------------------------
    # Property tests
    # ------------------------------------------------------------------

    @settings(max_examples=40, deadline=None)
    @given(
        theta=st.floats(min_value=0.05, max_value=jnp.pi - 0.05, allow_nan=False),
        radius=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    )
    def test_induced_sphere_at_phi_zero_always_diagonal(self, theta, radius):
        """InducedMetric on any sphere in sph2 at phi=0 is always exactly diagonal.

        At phi=0 the Jacobian columns are orthogonal without any cancellation:
        column_0 = [cos(θ), 0, -sin(θ)],  column_1 = [0, sin(θ), 0].
        Their dot-product is identically 0 in IEEE arithmetic.
        """
        manifold = cxm.EmbeddedManifold(
            intrinsic=cxm.HyperSphericalManifold(),
            ambient=cxm.EuclideanManifold(3),
            embed_map=cxm.TwoSphereIn3D(radius=radius),
        )
        p = {"theta": u.Angle(theta, "rad"), "phi": u.Angle(0.0, "rad")}
        result = manifold.metric.is_diagonal(cxc.sph2, at=p)
        assert bool(result) is True

    @settings(max_examples=40, deadline=None)
    @given(
        theta=st.floats(min_value=0.05, max_value=jnp.pi - 0.05, allow_nan=False),
        phi=st.floats(min_value=-jnp.pi, max_value=jnp.pi, allow_nan=False),
    )
    def test_is_diagonal_agrees_with_allclose_for_induced(self, theta, phi):
        """is_diagonal() must equal allclose(offdiag, 0) for InducedMetric at any point.

        The base implementation uses jnp.allclose (not exact equality).
        """
        metric = cxm.embedded_twosphere(radius=1.0).metric
        p = {"theta": u.Angle(theta, "rad"), "phi": u.Angle(phi, "rad")}
        result = metric.is_diagonal(cxc.sph2, at=p)
        G = metric.metric_matrix(cxc.sph2, at=p)
        val = G.value
        offdiag = val - jnp.diag(jnp.diag(val))
        expected = jnp.allclose(offdiag, jnp.zeros_like(offdiag))
        assert bool(result) == bool(expected)

    @settings(max_examples=40, deadline=None)
    @given(
        theta=st.floats(min_value=0.05, max_value=jnp.pi - 0.05, allow_nan=False),
        phi=st.floats(min_value=-jnp.pi, max_value=jnp.pi, allow_nan=False),
    )
    def test_abstractdiagonalmetric_matrix_is_nearly_diagonal(self, theta, phi):
        """AbstractDiagonalMetric structural promise: near-zero off-diagonals.

        is_diagonal() returns True unconditionally (structural promise), and the
        metric matrix confirms this: off-diagonal entries are within floating-point
        tolerance of zero even though is_diagonal doesn't evaluate the matrix.
        """
        metric = cxm.HyperSphericalMetric(ndim=2)
        assert isinstance(metric, cxm.AbstractDiagonalMetric)
        p = {"theta": u.Angle(theta, "rad"), "phi": u.Angle(phi, "rad")}
        # Structural promise (overridden method — no matrix evaluated)
        assert bool(metric.is_diagonal(cxc.sph2, at=p)) is True
        # Matrix-level confirmation
        G = metric.metric_matrix(cxc.sph2, at=p)
        offdiag = G.value - jnp.diag(jnp.diag(G.value))
        assert jnp.allclose(offdiag, jnp.zeros_like(offdiag), atol=1e-6)
