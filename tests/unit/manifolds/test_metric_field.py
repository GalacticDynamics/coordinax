"""Contract tests for AbstractMetricField concrete subtypes.

Verifies:
- All subtypes expose ``ndim`` and ``signature``
- Signature entries are ±1
- Static subtypes are JAX-static pytree leaves (no dynamic leaves)
- No ``metric_matrix``, ``scale_factors``, or ``cholesky`` methods exist
  (these were removed in Phase 3b)
- ``RoundMetric`` from ``metric/field.py`` is an eqx.Module with a dynamic
  ``radius`` leaf, static ``_ndim``, and supports JIT / grad through ``radius``
"""

import jax
import jax.numpy as jnp
import pytest

import unxt as u

import coordinax.manifolds as cxm
from coordinax._src.metric.field import (
    RoundMetric as DynamicRoundMetric,
)

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
            lambda: cxm.PullbackMetric(
                cxm.TwoSphereIn3D(radius=1.0), cxm.FlatMetric(3)
            ),
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


# ---------------------------------------------------------------------------
# RoundMetric from metric/field.py — dynamic radius, static ndim
# ---------------------------------------------------------------------------


class TestRoundMetricFieldDynamic:
    """RoundMetric (metric/field.py) is an eqx.Module with dynamic radius."""

    def test_radius_is_only_dynamic_leaf(self):
        m = DynamicRoundMetric(ndim=2, radius=u.Q(1.0, "m"))
        leaves, _ = jax.tree.flatten(m)
        assert len(leaves) == 1, f"Expected 1 dynamic leaf (radius), got {leaves}"

    def test_ndim_is_static(self):
        m2 = DynamicRoundMetric(ndim=2, radius=u.Q(1.0, "m"))
        m3 = DynamicRoundMetric(ndim=3, radius=u.Q(1.0, "m"))
        _, treedef2 = jax.tree.flatten(m2)
        _, treedef3 = jax.tree.flatten(m3)
        assert treedef2 != treedef3, (
            "ndim should be static (changing it should change the treedef)"
        )

    def test_signature_length_matches_ndim(self):
        for ndim in [1, 2, 3, 4]:
            m = DynamicRoundMetric(ndim=ndim, radius=u.Q(1.0, "m"))
            assert len(m.signature) == ndim
            assert all(s == 1 for s in m.signature)

    def test_jit_through_radius(self):
        m = DynamicRoundMetric(ndim=2, radius=u.Q(1.0, "m"))

        @jax.jit
        def get_radius_value(mf):
            return mf.radius.value

        result = get_radius_value(m)
        assert jnp.allclose(result, jnp.array(1.0))

    def test_grad_through_radius(self):
        def f(r_val):
            m = DynamicRoundMetric(ndim=2, radius=u.Q(r_val, "m"))
            return m.radius.value

        grad_f = jax.grad(f)
        result = grad_f(jnp.array(3.0))
        assert jnp.allclose(result, jnp.array(1.0))

    def test_radius_unit_preserved(self):
        m = DynamicRoundMetric(ndim=2, radius=u.Q(5.0, "km"))
        assert str(m.radius.unit) == "km"
        assert jnp.allclose(m.radius.value, jnp.array(5.0))
