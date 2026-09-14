"""Contract tests for AbstractMetricField concrete subtypes.

Verifies:
- All subtypes expose ``ndim`` and ``signature``
- Signature entries are ±1
- Static subtypes are JAX-static pytree leaves (no dynamic leaves)
- No ``metric_matrix``, ``scale_factors``, or ``cholesky`` methods exist
  (these were removed in Phase 3b)
"""

import jax
import pytest

import coordinax.manifolds as cxm

# ---------------------------------------------------------------------------
# Fixtures: every concrete AbstractMetricField subtype from the public API
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        pytest.param(lambda: cxm.FlatMetric(1), id="flat-1d"),
        pytest.param(lambda: cxm.FlatMetric(3), id="flat-3d"),
        pytest.param(lambda: cxm.RoundMetric(2), id="round-2d"),
        pytest.param(lambda: cxm.RoundMetric(3), id="round-3d"),
        pytest.param(lambda: cxm.MinkowskiMetric(), id="minkowski"),
        pytest.param(
            lambda: cxm.ProductMetric(factors=(cxm.RoundMetric(2), cxm.FlatMetric(1))),
            id="product-s2-r1",
        ),
        pytest.param(
            lambda: cxm.PullbackMetric(cxm.TwoSphereIn3D(radius=1), cxm.FlatMetric(3)),
            id="pullback-unit-sphere",
        ),
    ]
)
def metric_field(request):
    return request.param()


# ---------------------------------------------------------------------------
# Generic contract
# ---------------------------------------------------------------------------


class TestAbstractMetricFieldContract:
    """Every AbstractMetricField subtype satisfies these invariants."""

    def test_has_ndim(self, metric_field):
        assert isinstance(metric_field.ndim, int)
        assert metric_field.ndim >= 1

    def test_has_signature(self, metric_field):
        sig = metric_field.signature
        assert isinstance(sig, tuple)
        assert len(sig) == metric_field.ndim

    def test_signature_entries_are_plus_minus_one(self, metric_field):
        for s in metric_field.signature:
            assert s in (-1, 1), f"signature entry {s!r} is not ±1"

    def test_no_metric_matrix_method(self, metric_field):
        """Phase 3b: field classes must NOT have a metric_matrix() method."""
        assert not hasattr(metric_field, "metric_matrix"), (
            f"{type(metric_field).__name__} still has a metric_matrix method"
        )

    def test_no_scale_factors_method(self, metric_field):
        assert not hasattr(metric_field, "scale_factors"), (
            f"{type(metric_field).__name__} still has a scale_factors method"
        )

    def test_no_cholesky_method(self, metric_field):
        assert not hasattr(metric_field, "cholesky"), (
            f"{type(metric_field).__name__} still has a cholesky method"
        )


# ---------------------------------------------------------------------------
# Static JAX pytree leaves (parameter-free types)
# ---------------------------------------------------------------------------


class TestStaticMetricFieldPytree:
    """Parameter-free metric fields are static JAX pytrees (no dynamic leaves)."""

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: cxm.FlatMetric(3),
            lambda: cxm.RoundMetric(2),
            lambda: cxm.MinkowskiMetric(),
            lambda: cxm.ProductMetric(factors=(cxm.RoundMetric(2), cxm.FlatMetric(1))),
        ],
        ids=["flat-3d", "round-2d", "minkowski", "product"],
    )
    def test_no_dynamic_leaves(self, factory):
        m = factory()
        leaves, _ = jax.tree.flatten(m)
        assert leaves == [], (
            f"{type(m).__name__} has unexpected dynamic leaves: {leaves}"
        )

    def test_flat_metric_jit_roundtrip(self):
        m = cxm.FlatMetric(3)

        @jax.jit
        def get_ndim(mf):
            return mf.ndim

        assert get_ndim(m) == 3

    def test_round_metric_jit_roundtrip(self):
        m = cxm.RoundMetric(2)

        @jax.jit
        def get_ndim(mf):
            return mf.ndim

        assert get_ndim(m) == 2
