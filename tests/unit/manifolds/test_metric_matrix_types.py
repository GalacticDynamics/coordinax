"""Contract tests for `DiagonalMetric` and `DenseMetric`.

Both are metric-matrix representations with the same public surface -- `ndim`,
`to_dense`, `__matmul__`, `inverse`, `determinant`, and pytree/jit/vmap
behaviour. That surface is asserted once in `TestMetricContract`, against
`to_dense()` as the common reference, and parametrized over both types.

What is left per-class is what only that representation can say: that
`DiagonalMetric.to_dense()` is zero off the diagonal and inverts element-wise,
and that `DenseMetric.to_dense()` is the object itself.
"""

__all__: tuple[str, ...] = ()

import jax
import jax.numpy as jnp
import pytest

from coordinax._src.metric.matrix import DenseMetric, DiagonalMetric

# ---------------------------------------------------------------------------
# Fixtures


@pytest.fixture(
    params=[jnp.array([1, 4, 9]), jnp.array([1]), jnp.array([2, 3])],
    ids=["3d", "1d", "2d"],
)
def diag_metric(request):
    return DiagonalMetric(request.param)


@pytest.fixture(
    params=[jnp.eye(3), jnp.array([[4.0, 0.0], [0.0, 9.0]]), jnp.eye(1)],
    ids=["I3", "diag2", "I1"],
)
def dense_metric(request):
    return DenseMetric(request.param)


@pytest.fixture(
    params=[
        DiagonalMetric(jnp.array([1, 4, 9])),
        DiagonalMetric(jnp.array([1])),
        DiagonalMetric(jnp.array([2, 3])),
        DenseMetric(jnp.eye(3)),
        DenseMetric(jnp.array([[4.0, 0.0], [0.0, 9.0]])),
        DenseMetric(jnp.eye(1)),
    ],
    ids=["diag-3d", "diag-1d", "diag-2d", "dense-I3", "dense-diag2", "dense-I1"],
)
def metric(request):
    """Every metric representation, across every shape either is built with."""
    return request.param


# ---------------------------------------------------------------------------
# The shared surface


class TestMetricContract:
    """What both representations must do, checked against `to_dense()`."""

    def test_ndim_matches_dense_shape(self, metric) -> None:
        """The trailing two axes are (ndim, ndim); leading axes are batch.

        Only the trailing axes are constrained: `DiagonalMetric` supports a
        batched `(..., n)` diagonal, whose `to_dense()` is `(..., n, n)` --
        see `test_to_dense_batched_builds_diagonal`.
        """
        matrix = jnp.asarray(metric.to_dense().matrix)
        assert matrix.shape[-2:] == (metric.ndim, metric.ndim)

    def test_pytree_roundtrip(self, metric) -> None:
        """Flatten/unflatten recovers an equal object."""
        leaves, treedef = jax.tree_util.tree_flatten(metric)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(restored.to_dense().matrix, metric.to_dense().matrix)

    def test_is_a_dynamic_pytree(self, metric) -> None:
        """The matrix data is a JAX leaf, not the metric object itself.

        `len(leaves) > 0` alone would not show this: an unregistered object
        flattens to `[itself]`, one leaf, and would pass. So assert the leaf
        is an array and the metric is not its own leaf.
        """
        leaves, _ = jax.tree_util.tree_flatten(metric)
        assert leaves, "metric flattened to no leaves at all"
        assert metric not in leaves, "metric is an opaque leaf, not a pytree"
        assert all(isinstance(leaf, jnp.ndarray) for leaf in leaves)

    def test_matmul_agrees_with_dense(self, metric) -> None:
        v = jnp.arange(1, metric.ndim + 1, dtype=float)
        assert jnp.allclose(metric @ v, metric.to_dense().matrix @ v)

    def test_inverse_roundtrip_is_identity(self, metric) -> None:
        """G @ g^-1 == I."""
        product = metric.to_dense().matrix @ metric.inverse.to_dense().matrix
        assert jnp.allclose(product, jnp.eye(metric.ndim), atol=1e-5)

    def test_determinant_agrees_with_dense(self, metric) -> None:
        expected = jnp.linalg.det(metric.to_dense().matrix)
        assert jnp.allclose(metric.determinant, expected, atol=1e-5)

    def test_jit_matmul_agrees_with_eager(self, metric) -> None:
        v = jnp.ones(metric.ndim)

        @jax.jit
        def apply(g, v):
            return g @ v

        assert jnp.allclose(apply(metric, v), metric @ v)

    def test_jit_determinant_agrees_with_eager(self, metric) -> None:
        @jax.jit
        def det(g):
            return g.determinant

        assert jnp.allclose(det(metric), metric.determinant)

    def test_vmap_matmul(self, metric) -> None:
        """Vmap over a batch of vectors."""
        batch = jnp.ones((4, metric.ndim))
        result = jax.vmap(lambda v: metric @ v)(batch)
        assert result.shape == (4, metric.ndim)
        assert jnp.allclose(result[0], metric @ jnp.ones(metric.ndim))


# ---------------------------------------------------------------------------
# Representation-specific


class TestDiagonalMetric:
    """What only the diagonal representation can assert."""

    def test_ndim_is_the_diagonal_length(self, diag_metric) -> None:
        assert diag_metric.ndim == diag_metric.diagonal.shape[-1]

    def test_to_dense_puts_the_diagonal_on_the_diagonal(self, diag_metric) -> None:
        dense = diag_metric.to_dense()
        assert isinstance(dense, DenseMetric)
        assert jnp.allclose(jnp.diag(dense.matrix), diag_metric.diagonal)

    def test_to_dense_is_zero_off_the_diagonal(self, diag_metric) -> None:
        dense = diag_metric.to_dense()
        off_diagonal = ~jnp.eye(diag_metric.ndim, dtype=bool)
        assert jnp.allclose(dense.matrix[off_diagonal], 0)

    def test_to_dense_batched_builds_diagonal(self) -> None:
        """to_dense embeds a batched (B, n) diagonal into (B, n, n) matrices."""
        # jnp.diag would *extract* a diagonal from a batched array; must build one.
        diag = jnp.array([[1.0, 4.0, 9.0], [1.0, 16.0, 25.0]])  # (2, 3)
        dense = DiagonalMetric(diag).to_dense()
        m = jnp.asarray(dense.matrix)
        assert m.shape == (2, 3, 3)
        assert jnp.allclose(jnp.diagonal(m, axis1=-2, axis2=-1), diag)
        assert jnp.allclose(m[0][~jnp.eye(3, dtype=bool)], 0)

    def test_matmul_scales_componentwise(self, diag_metric) -> None:
        """G @ 1 is the diagonal itself."""
        result = diag_metric @ jnp.ones(diag_metric.ndim)
        assert jnp.allclose(result, diag_metric.diagonal)

    def test_inverse_stays_diagonal(self, diag_metric) -> None:
        inv = diag_metric.inverse
        assert isinstance(inv, DiagonalMetric)
        assert inv.diagonal.shape == diag_metric.diagonal.shape

    def test_inverse_is_reciprocal_elementwise(self, diag_metric) -> None:
        assert jnp.allclose(diag_metric.inverse.diagonal, 1 / diag_metric.diagonal)

    def test_determinant_is_the_product(self, diag_metric) -> None:
        assert jnp.allclose(diag_metric.determinant, jnp.prod(diag_metric.diagonal))


class TestDenseMetric:
    """What only the dense representation can assert."""

    def test_ndim_is_the_matrix_width(self, dense_metric) -> None:
        assert dense_metric.ndim == dense_metric.matrix.shape[-1]

    def test_to_dense_is_self(self, dense_metric) -> None:
        """No copy: the dense representation is already dense."""
        assert dense_metric.to_dense() is dense_metric


class TestDenseMetricKnownValues:
    """Closed-form anchors, so the contract is not only self-referential."""

    @pytest.mark.parametrize(
        ("matrix", "vector", "expected"),
        [
            pytest.param(jnp.eye(3), jnp.arange(1, 4), jnp.arange(1, 4), id="identity"),
            pytest.param(
                jnp.array([[4, 0], [0, 9]]),
                jnp.array([1, 1]),
                jnp.array([4, 9]),
                id="diagonal",
            ),
        ],
    )
    def test_matmul(self, matrix, vector, expected) -> None:
        assert jnp.allclose(DenseMetric(matrix) @ vector, expected)

    @pytest.mark.parametrize(
        ("matrix", "expected"),
        [
            pytest.param(jnp.eye(3), jnp.eye(3), id="identity"),
            pytest.param(
                jnp.array([[4, 0], [0, 9]]),
                jnp.array([[0.25, 0], [0, 1 / 9]]),
                id="diagonal",
            ),
        ],
    )
    def test_inverse(self, matrix, expected) -> None:
        assert jnp.allclose(DenseMetric(matrix).inverse.matrix, expected)

    @pytest.mark.parametrize(
        ("matrix", "expected"),
        [
            pytest.param(jnp.eye(3), 1, id="identity"),
            pytest.param(jnp.array([[2, 0], [0, 3]]), 6, id="diagonal"),
        ],
    )
    def test_determinant(self, matrix, expected) -> None:
        assert jnp.allclose(DenseMetric(matrix).determinant, expected)


# ---------------------------------------------------------------------------
# Cross-representation


class TestDiagonalDenseConsistency:
    """`DiagonalMetric.to_dense()` agrees with `DenseMetric` on operations."""

    def test_matmul_consistency(self) -> None:
        diag = DiagonalMetric(jnp.array([2, 3, 4]))
        v = jnp.array([1, 2, 3])
        assert jnp.allclose(diag @ v, diag.to_dense() @ v)

    def test_determinant_consistency(self) -> None:
        diag = DiagonalMetric(jnp.array([2, 3]))
        assert jnp.allclose(diag.determinant, diag.to_dense().determinant)

    def test_inverse_consistency(self) -> None:
        """Inverting then densifying equals densifying then inverting."""
        diag = DiagonalMetric(jnp.array([2, 5]))
        assert jnp.allclose(
            jnp.diag(diag.inverse.to_dense().matrix),
            diag.to_dense().inverse.matrix.diagonal(),
            atol=1e-6,
        )
