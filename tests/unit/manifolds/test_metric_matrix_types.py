"""Contract tests for DiagonalMetric and DenseMetric.

Tests cover:
- pytree flatten/unflatten round-trip (equinox Module)
- ``to_dense``
- ``__matmul__``
- ``inverse``
- ``determinant``
- JIT and vmap compatibility
"""

import jax
import jax.numpy as jnp
import pytest

from coordinax._src.metric.matrix import DenseMetric, DiagonalMetric

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        jnp.array([1.0, 4.0, 9.0]),
        jnp.array([1.0]),
        jnp.array([2.0, 3.0]),
    ],
    ids=["3d", "1d", "2d"],
)
def diag_metric(request):
    return DiagonalMetric(request.param)


@pytest.fixture(
    params=[
        jnp.eye(3),
        jnp.array([[4.0, 0.0], [0.0, 9.0]]),
        jnp.eye(1),
    ],
    ids=["I3", "diag2", "I1"],
)
def dense_metric(request):
    return DenseMetric(request.param)


# ---------------------------------------------------------------------------
# DiagonalMetric
# ---------------------------------------------------------------------------


class TestDiagonalMetric:
    """Contract tests for DiagonalMetric."""

    def test_ndim(self, diag_metric):
        assert diag_metric.ndim == diag_metric.diagonal.shape[-1]

    def test_pytree_roundtrip(self, diag_metric):
        """flatten/unflatten must recover an equal object."""
        leaves, treedef = jax.tree_util.tree_flatten(diag_metric)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(restored.diagonal, diag_metric.diagonal)

    def test_to_dense_shape(self, diag_metric):
        n = diag_metric.ndim
        dense = diag_metric.to_dense()
        assert isinstance(dense, DenseMetric)
        assert dense.matrix.shape == (n, n)

    def test_to_dense_diagonal_entries(self, diag_metric):
        """Diagonal entries of to_dense() match the stored diagonal."""
        dense = diag_metric.to_dense()
        assert jnp.allclose(jnp.diag(dense.matrix), diag_metric.diagonal)

    def test_to_dense_off_diagonal_zero(self, diag_metric):
        """Off-diagonal entries of to_dense() are zero."""
        n = diag_metric.ndim
        dense = diag_metric.to_dense()
        off_diag_mask = ~jnp.eye(n, dtype=bool)
        assert jnp.allclose(dense.matrix[off_diag_mask], 0.0)

    def test_matmul(self, diag_metric):
        n = diag_metric.ndim
        v = jnp.ones(n)
        result = diag_metric @ v
        assert jnp.allclose(result, diag_metric.diagonal)

    def test_matmul_matches_dense(self, diag_metric):
        n = diag_metric.ndim
        v = jnp.arange(1.0, n + 1)
        assert jnp.allclose(diag_metric @ v, diag_metric.to_dense() @ v)

    def test_inverse_shape(self, diag_metric):
        inv = diag_metric.inverse
        assert isinstance(inv, DiagonalMetric)
        assert inv.diagonal.shape == diag_metric.diagonal.shape

    def test_inverse_values(self, diag_metric):
        assert jnp.allclose(diag_metric.inverse.diagonal, 1.0 / diag_metric.diagonal)

    def test_inverse_roundtrip(self, diag_metric):
        """G * g⁻¹ ≈ I element-wise."""
        product = diag_metric.diagonal * diag_metric.inverse.diagonal
        assert jnp.allclose(product, jnp.ones(diag_metric.ndim))

    def test_determinant(self, diag_metric):
        assert jnp.allclose(diag_metric.determinant, jnp.prod(diag_metric.diagonal))

    def test_jit_matmul(self, diag_metric):
        n = diag_metric.ndim
        v = jnp.ones(n)

        @jax.jit
        def apply(d, v):
            return d @ v

        result = apply(diag_metric, v)
        assert jnp.allclose(result, diag_metric.diagonal)

    def test_jit_determinant(self, diag_metric):
        @jax.jit
        def det(d):
            return d.determinant

        assert jnp.allclose(det(diag_metric), diag_metric.determinant)

    def test_vmap_matmul(self, diag_metric):
        """Vmap over batch of vectors."""
        n = diag_metric.ndim
        batch = jnp.ones((4, n))

        result = jax.vmap(lambda v: diag_metric @ v)(batch)
        assert result.shape == (4, n)
        assert jnp.allclose(result[0], diag_metric.diagonal)

    def test_is_not_static(self):
        """DiagonalMetric is a dynamic pytree, not a static leaf."""
        d = DiagonalMetric(jnp.array([1.0, 2.0]))
        leaves, _ = jax.tree_util.tree_flatten(d)
        assert len(leaves) > 0, "diagonal should be a dynamic JAX leaf"


# ---------------------------------------------------------------------------
# DenseMetric
# ---------------------------------------------------------------------------


class TestDenseMetric:
    """Contract tests for DenseMetric."""

    def test_ndim(self, dense_metric):
        assert dense_metric.ndim == dense_metric.matrix.shape[-1]

    def test_pytree_roundtrip(self, dense_metric):
        leaves, treedef = jax.tree_util.tree_flatten(dense_metric)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(restored.matrix, dense_metric.matrix)

    def test_to_dense_is_self(self, dense_metric):
        assert dense_metric.to_dense() is dense_metric

    def test_matmul_identity(self):
        n = 3
        g = DenseMetric(jnp.eye(n))
        v = jnp.arange(1.0, n + 1)
        assert jnp.allclose(g @ v, v)

    def test_matmul_diagonal(self):
        g = DenseMetric(jnp.array([[4.0, 0.0], [0.0, 9.0]]))
        v = jnp.array([1.0, 1.0])
        assert jnp.allclose(g @ v, jnp.array([4.0, 9.0]))

    def test_inverse_identity(self):
        g = DenseMetric(jnp.eye(3))
        assert jnp.allclose(g.inverse.matrix, jnp.eye(3))

    def test_inverse_diagonal(self):
        g = DenseMetric(jnp.array([[4.0, 0.0], [0.0, 9.0]]))
        expected = jnp.array([[0.25, 0.0], [0.0, 1.0 / 9.0]])
        assert jnp.allclose(g.inverse.matrix, expected)

    def test_inverse_roundtrip(self, dense_metric):
        n = dense_metric.ndim
        product = dense_metric.matrix @ dense_metric.inverse.matrix
        assert jnp.allclose(product, jnp.eye(n), atol=1e-5)

    def test_determinant_identity(self):
        assert jnp.allclose(DenseMetric(jnp.eye(3)).determinant, 1.0)

    def test_determinant_diagonal(self):
        g = DenseMetric(jnp.array([[2.0, 0.0], [0.0, 3.0]]))
        assert jnp.allclose(g.determinant, 6.0)

    def test_jit_matmul(self, dense_metric):
        n = dense_metric.ndim
        v = jnp.ones(n)

        @jax.jit
        def apply(g, v):
            return g @ v

        result = apply(dense_metric, v)
        assert result.shape == (n,)

    def test_jit_determinant(self, dense_metric):
        @jax.jit
        def det(g):
            return g.determinant

        assert jnp.isfinite(det(dense_metric))

    def test_vmap_matmul(self, dense_metric):
        n = dense_metric.ndim
        batch = jnp.ones((4, n))

        result = jax.vmap(lambda v: dense_metric @ v)(batch)
        assert result.shape == (4, n)

    def test_is_not_static(self):
        g = DenseMetric(jnp.eye(2))
        leaves, _ = jax.tree_util.tree_flatten(g)
        assert len(leaves) > 0


# ---------------------------------------------------------------------------
# DiagonalMetric ↔ DenseMetric consistency
# ---------------------------------------------------------------------------


class TestDiagonalDenseConsistency:
    """DiagonalMetric.to_dense() must agree with DenseMetric on operations."""

    def test_matmul_consistency(self):
        diag = DiagonalMetric(jnp.array([2.0, 3.0, 4.0]))
        dense = diag.to_dense()
        v = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(diag @ v, dense @ v)

    def test_determinant_consistency(self):
        diag = DiagonalMetric(jnp.array([2.0, 3.0]))
        dense = diag.to_dense()
        assert jnp.allclose(diag.determinant, dense.determinant)

    def test_inverse_consistency(self):
        diag = DiagonalMetric(jnp.array([2.0, 5.0]))
        assert jnp.allclose(
            jnp.diag(diag.inverse.to_dense().matrix),
            diag.to_dense().inverse.matrix.diagonal(),
            atol=1e-6,
        )
