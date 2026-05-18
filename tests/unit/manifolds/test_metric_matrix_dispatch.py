"""Tests for metric_matrix() and metric_representation() dispatch rules.

Coverage:
- All registered (manifold, chart) pairs return the type promised by
  ``metric_representation(manifold, chart)``
- Numerical values at sample points for the constant-metric cases
- JIT compatibility for each dispatch path
"""

import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.api.manifolds as cxmapi
import coordinax.charts as cxc
import coordinax.manifolds as cxm
from coordinax._src.metric.matrix import DiagonalMetric

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_product_manifold():
    """Return a CartesianProductManifold (S² x R¹) and its default chart."""
    M_prod = cxm.CartesianProductManifold(
        factors=(cxm.S2, cxm.R1), factor_names=("sphere", "line")
    )
    chart = M_prod.atlas.default_chart()
    return M_prod, chart


# ---------------------------------------------------------------------------
# Fixtures: (manifold, point, chart) triples
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        pytest.param("euclidean_cart1d", id="euclidean-cart1d"),
        pytest.param("euclidean_cart2d", id="euclidean-cart2d"),
        pytest.param("euclidean_cart3d", id="euclidean-cart3d"),
        pytest.param("euclidean_cartnd", id="euclidean-cartnd"),
        pytest.param("euclidean_sph3d", id="euclidean-sph3d"),
        pytest.param("hyperspherical_sph2", id="hyperspherical-sph2"),
        pytest.param("minkowski_minkowskict", id="minkowski-minkowskict"),
        pytest.param("product_default", id="product-default"),
    ]
)
def manifold_point_chart(request):
    """Return ``(manifold, point_dict, chart)`` for each registered pair."""
    key = request.param
    if key == "euclidean_cart1d":
        return (cxm.R1, {"x": jnp.array(1)}, cxc.cart1d)
    if key == "euclidean_cart2d":
        return (cxm.R2, {"x": jnp.array(1), "y": jnp.array(2)}, cxc.cart2d)
    if key == "euclidean_cart3d":
        return (
            cxm.R3,
            {"x": jnp.array(1), "y": jnp.array(2), "z": jnp.array(3)},
            cxc.cart3d,
        )
    if key == "euclidean_cartnd":
        return (cxm.R3, {"q": jnp.array([1, 2, 3])}, cxc.cartnd)
    if key == "euclidean_sph3d":
        return (
            cxm.R3,
            {
                "r": u.Q(jnp.array(2), "m"),
                "theta": u.Angle(jnp.pi / 3, "rad"),
                "phi": u.Angle(jnp.array(0.4), "rad"),
            },
            cxc.sph3d,
        )
    if key == "hyperspherical_sph2":
        return (cxm.S2, {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0)}, cxc.sph2)
    if key == "minkowski_minkowskict":
        return (
            cxm.MinkowskiManifold(),
            {
                "ct": jnp.array(0),
                "x": jnp.array(1),
                "y": jnp.array(0),
                "z": jnp.array(0),
            },
            cxc.minkowskict,
        )
    if key == "product_default":
        M_prod, chart = _make_product_manifold()
        pt = {k: jnp.array(jnp.pi / 4) for k in chart.components}
        return (M_prod, pt, chart)
    msg = f"Unknown fixture key: {key!r}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Contract tests — apply to every registered (manifold, chart) pair
# ---------------------------------------------------------------------------


class TestMetricMatrixDispatchContract:
    """Each registered (manifold, chart) pair must satisfy these invariants."""

    def test_returns_correct_type(self, manifold_point_chart):
        """``metric_matrix`` must return an instance of ``metric_representation``."""
        manifold, point, chart = manifold_point_chart
        expected_cls = cxmapi.metric_representation(manifold, chart)
        result = cxmapi.metric_matrix(manifold, point, chart)
        assert isinstance(result, expected_cls), (
            f"metric_matrix({type(manifold).__name__}, {type(chart).__name__}) "
            f"returned {type(result).__name__!r}, expected {expected_cls.__name__!r}"
        )

    def test_metric_representation_returns_type(self, manifold_point_chart):
        """``metric_representation`` must return a class, not an instance."""
        manifold, _point, chart = manifold_point_chart
        cls = cxmapi.metric_representation(manifold, chart)
        assert isinstance(cls, type), (
            f"metric_representation should return a class, got {type(cls).__name__!r}"
        )

    def test_has_ndim(self, manifold_point_chart):
        """Returned matrix must have a positive integer ndim."""
        manifold, point, chart = manifold_point_chart
        result = cxmapi.metric_matrix(manifold, point, chart)
        assert hasattr(result, "ndim")
        assert isinstance(result.ndim, int)
        assert result.ndim > 0


# ---------------------------------------------------------------------------
# Numerical value tests
# ---------------------------------------------------------------------------


class TestMetricMatrixNumericalValues:
    """Spot-check numerical values for constant-metric cases."""

    def test_euclidean_cart1d_is_identity(self):
        pt = {"x": jnp.array(0)}
        g = cxmapi.metric_matrix(cxm.R1, pt, cxc.cart1d)
        assert isinstance(g, cxm.DiagonalMetric)
        assert jnp.allclose(g.diagonal, jnp.ones(1))

    def test_euclidean_cart2d_is_identity(self):
        pt = {"x": jnp.array(0), "y": jnp.array(0)}
        g = cxmapi.metric_matrix(cxm.R2, pt, cxc.cart2d)
        assert isinstance(g, cxm.DiagonalMetric)
        assert jnp.allclose(g.diagonal, jnp.ones(2))

    def test_euclidean_cart3d_is_identity(self):
        pt = {"x": jnp.array(1), "y": jnp.array(2), "z": jnp.array(3)}
        g = cxmapi.metric_matrix(cxm.R3, pt, cxc.cart3d)
        assert isinstance(g, cxm.DiagonalMetric)
        assert jnp.allclose(g.diagonal, jnp.ones(3))

    def test_euclidean_cartnd_identity_by_dimension(self):
        """CartND: diagonal length equals the actual array dimensionality."""
        pt = {"q": jnp.array([1, 2, 3])}
        g = cxmapi.metric_matrix(cxm.R3, pt, cxc.cartnd)
        assert isinstance(g, cxm.DiagonalMetric)
        assert g.diagonal.shape == (3,)
        assert jnp.allclose(g.diagonal, jnp.ones(3))

    def test_minkowski_diagonal_signature(self):
        """Minkowski metric in (ct, x, y, z) coords: diag = [-1, 1, 1, 1]."""
        M = cxm.MinkowskiManifold()
        pt = {
            "ct": jnp.array(0),
            "x": jnp.array(1),
            "y": jnp.array(0),
            "z": jnp.array(0),
        }
        g = cxmapi.metric_matrix(M, pt, cxc.minkowskict)
        assert isinstance(g, DiagonalMetric)
        assert jnp.allclose(g.diagonal, jnp.array([-1, 1, 1, 1]))

    def test_hyperspherical_at_equator(self):
        """At theta=π/2: diag(S²) = (1, sin²(π/2)) = (1, 1)."""
        pt = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0)}
        g = cxmapi.metric_matrix(cxm.S2, pt, cxc.sph2)
        assert isinstance(g, DiagonalMetric)
        assert jnp.allclose(g.diagonal, jnp.array([1, 1]), atol=1e-6)

    def test_hyperspherical_at_off_equator(self):
        """At theta=π/3: diag(S²) = (1, sin²(π/3)) = (1, 3/4)."""
        pt = {"theta": jnp.array(jnp.pi / 3), "phi": jnp.array(0)}
        g = cxmapi.metric_matrix(cxm.S2, pt, cxc.sph2)
        assert isinstance(g, DiagonalMetric)
        expected = jnp.array([1, jnp.sin(jnp.pi / 3) ** 2])
        assert jnp.allclose(g.diagonal, expected, atol=1e-6)

    def test_euclidean_sph3d_returns_diagonal_metric(self):
        """EuclideanManifold + sph3d uses analytic formula, returning DiagonalMetric."""
        pt = {
            "r": u.Q(jnp.array(2), "m"),
            "theta": u.Angle(jnp.pi / 2, "rad"),
            "phi": u.Angle(jnp.array(0), "rad"),
        }
        g = cxmapi.metric_matrix(cxm.R3, pt, cxc.sph3d)
        assert isinstance(g, DiagonalMetric)

    def test_product_manifold_returns_dense_metric(self):
        """CartesianProductManifold always returns DenseMetric."""
        M_prod, chart = _make_product_manifold()
        pt = {k: jnp.array(jnp.pi / 4) for k in chart.components}
        g = cxmapi.metric_matrix(M_prod, pt, chart)
        assert isinstance(g, cxm.DenseMetric)


# ---------------------------------------------------------------------------
# DiagonalMetric consistency
# ---------------------------------------------------------------------------


class TestDiagonalMetricOffDiagonal:
    """DiagonalMetric.to_dense() must have zero off-diagonal entries."""

    def test_euclidean_cart3d_dense_is_identity(self):
        pt = {"x": jnp.array(1), "y": jnp.array(2), "z": jnp.array(3)}
        g = cxmapi.metric_matrix(cxm.R3, pt, cxc.cart3d)
        assert isinstance(g, DiagonalMetric)
        G = g.to_dense().matrix
        off_diag = G - jnp.diag(jnp.diag(G))
        assert jnp.allclose(off_diag, jnp.zeros((3, 3)))

    def test_minkowski_dense_off_diagonal_is_zero(self):
        M = cxm.MinkowskiManifold()
        pt = {
            "ct": jnp.array(0),
            "x": jnp.array(0),
            "y": jnp.array(0),
            "z": jnp.array(0),
        }
        g = cxmapi.metric_matrix(M, pt, cxc.minkowskict)
        assert isinstance(g, DiagonalMetric)
        G = g.to_dense().matrix
        off_diag = G - jnp.diag(jnp.diag(G))
        assert jnp.allclose(off_diag, jnp.zeros((4, 4)))


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


class TestMetricMatrixJIT:
    """metric_matrix must be JIT-compilable for all dispatch paths."""

    def test_jit_euclidean_cart3d(self):
        @jax.jit
        def compute(x, y, z):
            pt = {"x": x, "y": y, "z": z}
            return cxmapi.metric_matrix(cxm.R3, pt, cxc.cart3d).diagonal

        result = compute(jnp.array(1), jnp.array(2), jnp.array(3))
        assert jnp.allclose(result, jnp.ones(3))

    def test_jit_euclidean_cartnd(self):
        @jax.jit
        def compute(q):
            return cxmapi.metric_matrix(cxm.R3, {"q": q}, cxc.cartnd).diagonal

        result = compute(jnp.array([1, 2, 3]))
        assert jnp.allclose(result, jnp.ones(3))

    def test_jit_minkowski(self):
        M = cxm.MinkowskiManifold()
        chart = cxc.minkowskict

        @jax.jit
        def compute(ct, x, y, z):
            pt = {"ct": ct, "x": x, "y": y, "z": z}
            return cxmapi.metric_matrix(M, pt, chart).diagonal

        result = compute(jnp.array(0), jnp.array(0), jnp.array(0), jnp.array(0))
        assert jnp.allclose(result, jnp.array([-1, 1, 1, 1]))

    def test_jit_hyperspherical(self):
        @jax.jit
        def compute(theta, phi):
            return cxmapi.metric_matrix(
                cxm.S2, {"theta": theta, "phi": phi}, cxc.sph2
            ).diagonal

        result = compute(jnp.array(jnp.pi / 2), jnp.array(0))
        assert jnp.allclose(result, jnp.array([1, 1]), atol=1e-6)

    def test_jit_cart1d(self):
        @jax.jit
        def compute(x):
            return cxmapi.metric_matrix(cxm.R1, {"x": x}, cxc.cart1d).diagonal

        result = compute(jnp.array(0))
        assert jnp.allclose(result, jnp.ones(1))

    def test_jit_cart2d(self):
        @jax.jit
        def compute(x, y):
            return cxmapi.metric_matrix(cxm.R2, {"x": x, "y": y}, cxc.cart2d).diagonal

        result = compute(jnp.array(0), jnp.array(0))
        assert jnp.allclose(result, jnp.ones(2))


# ---------------------------------------------------------------------------
# metric_representation returns constant result (not point-dependent)
# ---------------------------------------------------------------------------


class TestMetricRepresentation:
    """metric_representation must return the correct class for each pair."""

    @pytest.mark.parametrize(
        ("manifold", "chart", "expected_cls"),
        [
            (cxm.R1, cxc.cart1d, cxm.DiagonalMetric),
            (cxm.R2, cxc.cart2d, cxm.DiagonalMetric),
            (cxm.R3, cxc.cart3d, cxm.DiagonalMetric),
            (cxm.R3, cxc.cartnd, cxm.DiagonalMetric),
            (cxm.R3, cxc.sph3d, cxm.DiagonalMetric),
            (cxm.S2, cxc.sph2, cxm.DiagonalMetric),
            (cxm.MinkowskiManifold(), cxc.minkowskict, cxm.DiagonalMetric),
        ],
    )
    def test_metric_representation_type(self, manifold, chart, expected_cls):
        assert cxmapi.metric_representation(manifold, chart) is expected_cls
