"""Tests for coordinax.metrics — AbstractMetricField and concrete implementations.

All tests in this file are RED until the metrics module is implemented.
"""

import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.charts as cxc
import coordinax.manifolds as cxm
from coordinax._src.metric.matrix import DenseMetric, DiagonalMetric
from coordinax.api.manifolds import metric_matrix as mm_dispatch
from coordinax.internal import QMatrix


def _mat_val(dense_metric, /):
    """Extract numeric array from DenseMetric, regardless of matrix type."""
    mat = dense_metric.matrix
    return mat.value if isinstance(mat, QMatrix) else mat


# =============================================================================
# AbstractMetricField contract
# =============================================================================


class TestAbstractMetricFieldContract:
    """Every AbstractMetricField subclass must satisfy these invariants."""

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
            "R3": cxm.FlatMetric(3),
            "R2": cxm.FlatMetric(2),
            "R1": cxm.FlatMetric(1),
            "hyperspherical2d": cxm.RoundMetric(ndim=2),
            "minkowski4d": cxm.MinkowskiMetric(),
            "product3d": cxm.ProductMetric(
                factors=(cxm.RoundMetric(2), cxm.FlatMetric(1))
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
# AbstractDiagonalMetricField contract
# =============================================================================


class TestAbstractDiagonalMetricFieldContract:
    """Every AbstractDiagonalMetricField subclass must report is_diagonal=True.

    This class tests the structural promise added by AbstractDiagonalMetricField beyond
    what AbstractMetricField already guarantees: is_diagonal() must return True for all
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
                cxm.FlatMetric(3),
                cxc.cart3d,
                {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")},
            ),
            "R3_sph": (
                cxm.FlatMetric(3),
                cxc.sph3d,
                {
                    "r": u.Q(2.0, "m"),
                    "theta": u.Angle(jnp.pi / 3, "rad"),
                    "phi": u.Angle(1.0, "rad"),
                },
            ),
            "R2_cart": (
                cxm.FlatMetric(2),
                cxc.cart2d,
                {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")},
            ),
            "hyperspherical2d": (
                cxm.RoundMetric(ndim=2),
                cxc.sph2,
                {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")},
            ),
            "minkowski4d": (
                cxm.MinkowskiMetric(),
                cxc.MinkowskiCT(),
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
        assert isinstance(metric, cxm.AbstractDiagonalMetricField)

    def test_is_diagonal_returns_true(self, metric_chart_at):
        """AbstractDiagonalMetricField subclasses are always diagonal by type."""
        metric, _, _ = metric_chart_at
        assert isinstance(metric, cxm.AbstractDiagonalMetricField)

    def test_is_diagonal_output_shape(self, metric_chart_at):
        """AbstractDiagonalMetricField structural check (no method call needed)."""
        metric, _, _ = metric_chart_at
        assert isinstance(metric, cxm.AbstractDiagonalMetricField)

    def test_is_diagonal_under_jit(self, metric_chart_at):
        """Isinstance check doesn't require jit — type is static."""
        metric, _, _ = metric_chart_at
        assert isinstance(metric, cxm.AbstractDiagonalMetricField)

    def test_is_diagonal_ignores_usys(self, metric_chart_at):
        """Diagonal-ness is a structural property independent of usys."""
        metric, _, _ = metric_chart_at
        assert isinstance(metric, cxm.AbstractDiagonalMetricField)


# =============================================================================
# FlatMetric
# =============================================================================


class TestFlatMetric:
    """Tests for FlatMetric, the flat Riemannian metric on Euclidean space."""

    def test_isinstance_abstractdiagonalmetric(self):
        assert isinstance(cxm.FlatMetric(3), cxm.AbstractDiagonalMetricField)

    def test_construction_1d(self):
        m = cxm.FlatMetric(1)
        assert m.ndim == 1

    def test_construction_2d(self):
        m = cxm.FlatMetric(2)
        assert m.ndim == 2

    def test_construction_3d(self):
        m = cxm.FlatMetric(3)
        assert m.ndim == 3

    def test_metric_matrix_cart3d_is_identity(self):
        p = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}
        g = mm_dispatch(cxm.R3, p, cxc.cart3d)
        assert isinstance(g, DiagonalMetric)
        assert jnp.allclose(g.diagonal, jnp.ones(3))

    def test_metric_matrix_cart2d_is_identity(self):
        p = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m")}
        g = mm_dispatch(cxm.R2, p, cxc.cart2d)
        assert isinstance(g, DiagonalMetric)
        assert jnp.allclose(g.diagonal, jnp.ones(2))

    def test_metric_matrix_sph3d_at_origin(self):
        """Spherical metric at (1, pi/2, 0): diag(1, r^2, r^2 sin^2 theta)."""
        p = {
            "r": u.Q(1.0, "m"),
            "theta": u.Angle(jnp.pi / 2, "rad"),
            "phi": u.Angle(0.0, "rad"),
        }
        g = mm_dispatch(cxm.R3, p, cxc.sph3d)
        dense = g.to_dense()
        assert dense.matrix.shape == (3, 3)
        # diagonal entries: g_rr=1, g_tt=r^2=1, g_pp=r^2 sin^2(theta)=1
        expected_diag = jnp.array([1.0, 1.0, 1.0])
        assert jnp.allclose(jnp.diag(_mat_val(dense)), expected_diag, atol=1e-6)

    def test_metric_matrix_sph3d_diagonal(self):
        """Spherical metric is always diagonal."""
        p = {
            "r": u.Q(2.0, "m"),
            "theta": u.Angle(jnp.pi / 3, "rad"),
            "phi": u.Angle(1.0, "rad"),
        }
        g = mm_dispatch(cxm.R3, p, cxc.sph3d)
        assert isinstance(g, DiagonalMetric)

    def test_metric_matrix_jit(self):
        p = {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")}

        @jax.jit
        def compute(p):
            return mm_dispatch(cxm.R3, p, cxc.cart3d)

        g = compute(p)
        assert isinstance(g, DiagonalMetric)

    def test_carried_by_euclidean_manifold(self):
        """R3.metric should return an FlatMetric."""
        metric = cxm.R3.metric
        assert isinstance(metric, cxm.FlatMetric)
        assert metric.ndim == 3


# =============================================================================
# RoundMetric
# =============================================================================


class TestRoundMetric:
    """Tests for RoundMetric, the round metric on the unit sphere."""

    def test_isinstance_abstractdiagonalmetric(self):
        assert isinstance(cxm.RoundMetric(ndim=2), cxm.AbstractDiagonalMetricField)

    def test_construction(self):
        m = cxm.RoundMetric(ndim=2)
        assert m.ndim == 2

    def test_metric_matrix_at_equator(self):
        """S^2 metric at equator: diag(1, sin^2(theta)) = diag(1, 1)."""
        p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
        g = mm_dispatch(cxm.S2, p, cxc.sph2)
        assert isinstance(g, DiagonalMetric)
        expected = jnp.array([1.0, 1.0])
        assert jnp.allclose(g.diagonal, expected, atol=1e-6)

    def test_metric_matrix_at_pole_theta_component(self):
        """S^2 metric g_theta_theta = 1 everywhere."""
        p = {"theta": u.Angle(0.1, "rad"), "phi": u.Angle(0.0, "rad")}
        g = mm_dispatch(cxm.S2, p, cxc.sph2)
        assert jnp.allclose(g.diagonal[0], 1.0, atol=1e-6)

    @pytest.mark.parametrize("theta", [0.1, jnp.pi / 4, jnp.pi / 2, jnp.pi * 3 / 4])
    def test_metric_matrix_phi_component_at_various_latitudes(self, theta):
        """g_phi_phi = sin^2(theta)."""
        p = {"theta": u.Angle(theta, "rad"), "phi": u.Angle(0.0, "rad")}
        g = mm_dispatch(cxm.S2, p, cxc.sph2)
        exp = jnp.sin(theta) ** 2
        assert jnp.allclose(g.diagonal[1], exp, atol=1e-6), (
            f"theta={theta}: g_phi_phi={g.diagonal[1]} != sin^2(theta)={exp}"
        )

    def test_metric_matrix_is_diagonal(self):
        """S^2 metric matrix is always diagonal."""
        p = {"theta": u.Angle(jnp.pi / 3, "rad"), "phi": u.Angle(1.0, "rad")}
        g = mm_dispatch(cxm.S2, p, cxc.sph2)
        assert isinstance(g, DiagonalMetric)

    def test_metric_matrix_jit(self):
        p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}

        @jax.jit
        def compute(p):
            return mm_dispatch(cxm.S2, p, cxc.sph2)

        g = compute(p)
        assert isinstance(g, DiagonalMetric)

    def test_metric_matrix_vmap(self):
        thetas = jnp.linspace(0.1, jnp.pi - 0.1, 5)

        def single(theta_val):
            p = {"theta": u.Angle(theta_val, "rad"), "phi": u.Angle(0.0, "rad")}
            return mm_dispatch(cxm.S2, p, cxc.sph2)

        gs = jax.vmap(single)(thetas)
        assert gs.diagonal.shape == (5, 2)

    def test_carried_by_hyperspherical_manifold(self):
        """S2.metric should return a RoundMetric."""
        metric = cxm.S2.metric
        assert isinstance(metric, cxm.RoundMetric)
        assert metric.ndim == 2


# =============================================================================
# MinkowskiMetric
# =============================================================================


class TestMinkowskiMetric:
    """Tests for MinkowskiMetric, the flat Lorentzian metric on Minkowski spacetime."""

    def test_isinstance_abstractdiagonalmetric(self):
        assert isinstance(cxm.MinkowskiMetric(), cxm.AbstractDiagonalMetricField)

    def test_construction(self):
        M = cxm.MinkowskiMetric()
        assert M.ndim == 4

    def test_metric_matrix_is_diagonal(self):
        chart = cxc.MinkowskiCT()
        p = {
            "ct": u.Q(1.0, "m"),
            "x": u.Q(0.0, "m"),
            "y": u.Q(0.0, "m"),
            "z": u.Q(0.0, "m"),
        }
        g = mm_dispatch(cxm.MinkowskiManifold(), p, chart)
        assert isinstance(g, DiagonalMetric)
        expected = jnp.array([-1.0, 1.0, 1.0, 1.0])
        assert jnp.allclose(g.diagonal, expected, atol=1e-6)

    def test_metric_matrix_is_position_independent(self):
        """Minkowski metric is flat — does not depend on position."""
        chart = cxc.MinkowskiCT()
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
        g1 = mm_dispatch(cxm.MinkowskiManifold(), p1, chart)
        g2 = mm_dispatch(cxm.MinkowskiManifold(), p2, chart)
        assert jnp.allclose(g1.diagonal, g2.diagonal, atol=1e-6)

    def test_metric_matrix_jit(self):
        chart = cxc.MinkowskiCT()
        p = {
            "ct": u.Q(1.0, "m"),
            "x": u.Q(0.0, "m"),
            "y": u.Q(0, "m"),
            "z": u.Q(0, "m"),
        }

        @jax.jit
        def compute(p):
            return mm_dispatch(cxm.MinkowskiManifold(), p, chart)

        g = compute(p)
        assert isinstance(g, DiagonalMetric)

    def test_carried_by_minkowski_manifold(self):
        """minkowski4d.metric should return a MinkowskiMetric."""
        metric = cxm.minkowski4d.metric
        assert isinstance(metric, cxm.MinkowskiMetric)
        assert metric.ndim == 4


# =============================================================================
# PullbackMetric
# =============================================================================


class TestPullbackMetric:
    """Tests for PullbackMetric, the pullback metric on an embedded manifold."""

    def test_induced_metric_is_not_abstractdiagonalmetric(self):
        """PullbackMetric is NOT an AbstractDiagonalMetricField."""
        manifold = cxm.embedded_twosphere(radius=1.0)
        assert not isinstance(manifold.metric, cxm.AbstractDiagonalMetricField)

    def test_unit_sphere_at_equator(self):
        """Induced metric on S^2 embedded in R^3 at equator matches sphere metric."""
        manifold = cxm.embedded_twosphere(radius=1.0)
        assert isinstance(manifold.metric, cxm.PullbackMetric)

        p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
        g = mm_dispatch(manifold, p, cxc.sph2)
        assert isinstance(g, DenseMetric)
        assert g.matrix.shape == (2, 2)
        # At equator: diag(1, sin^2(pi/2)) = diag(1, 1)
        expected = jnp.eye(2)
        assert jnp.allclose(_mat_val(g), expected, atol=1e-4)

    def test_radius_2_sphere_at_equator(self):
        """Induced metric on radius-2 sphere.

        diag(R^2, R^2 sin^2 theta) at equator = diag(4, 4).
        """
        manifold = cxm.EmbeddedManifold(
            intrinsic=cxm.S2,
            ambient=cxm.R3,
            embed_map=cxm.TwoSphereIn3D(radius=u.Q(2.0, "m")),
        )
        assert isinstance(manifold.metric, cxm.PullbackMetric)

        p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
        g = mm_dispatch(manifold, p, cxc.sph2)
        assert isinstance(g, DenseMetric)
        assert g.matrix.shape == (2, 2)
        expected = jnp.diag(jnp.array([4.0, 4.0]))
        assert jnp.allclose(_mat_val(g), expected, atol=1e-3)

    def test_induced_metric_jit(self):
        """metric_matrix dispatch should work under jit for EmbeddedManifold."""
        manifold = cxm.embedded_twosphere(radius=1.0)

        @jax.jit
        def compute(p):
            return mm_dispatch(manifold, p, cxc.sph2)

        p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
        g = compute(p)
        assert isinstance(g, DenseMetric)
        assert g.matrix.shape == (2, 2)


# =============================================================================
# ProductMetric
# =============================================================================


class TestProductMetric:
    """Tests for ProductMetric on product manifolds."""

    def test_product_metric_is_not_abstractdiagonalmetric(self):
        """ProductMetric is NOT an AbstractDiagonalMetricField."""
        M = cxm.CartesianProductManifold(
            factors=(cxm.S2, cxm.R1), factor_names=("S2", "R1")
        )
        assert not isinstance(M.metric, cxm.AbstractDiagonalMetricField)

    def test_signature_concatenates_factor_signatures(self):
        M = cxm.CartesianProductManifold(
            factors=(cxm.MinkowskiManifold(), cxm.R1), factor_names=("st", "x")
        )
        metric = M.metric
        assert isinstance(metric, cxm.ProductMetric)
        assert metric.signature == (-1, 1, 1, 1, 1)
        assert metric.ndim == 5

    def test_metric_matrix_is_block_diagonal_in_product_chart(self):
        M = cxm.CartesianProductManifold(
            factors=(cxm.S2, cxm.R1), factor_names=("S2", "R1")
        )
        chart = cxc.CartesianProductChart((cxc.sph2, cxc.cart1d), ("S2", "R1"))
        p = {
            "S2.theta": u.Angle(jnp.pi / 2, "rad"),
            "S2.phi": u.Angle(0.0, "rad"),
            "R1.x": u.Q(1.0, "m"),
        }

        g = mm_dispatch(M, p, chart)
        assert isinstance(g, DenseMetric)
        assert g.matrix.shape == (3, 3)
        assert jnp.allclose(_mat_val(g), jnp.eye(3), atol=1e-6)

    def test_product_manifold_carries_product_metric(self):
        M = cxm.CartesianProductManifold(
            factors=(cxm.S2, cxm.R1), factor_names=("S2", "R1")
        )
        assert isinstance(M.metric, cxm.ProductMetric)
