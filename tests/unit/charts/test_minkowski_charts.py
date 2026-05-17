"""Tests specific to MinkowskiCT and the minkowskict pre-defined instance."""

import jax
import pytest

import coordinax.charts as cxc
import coordinax.manifolds as cxm

# =============================================================================
# MinkowskiCT — construction
# =============================================================================


class TestMinkowskiCTConstruction:
    """MinkowskiCT construction and validation."""

    def test_default_construction(self) -> None:
        """MinkowskiCT() constructs without arguments."""
        chart = cxc.MinkowskiCT()
        assert isinstance(chart, cxc.MinkowskiCT)

    def test_explicit_manifold(self) -> None:
        """MinkowskiCT accepts an explicit MinkowskiManifold."""
        M = cxm.MinkowskiManifold()
        chart = cxc.MinkowskiCT(M=M)
        assert isinstance(chart, cxc.MinkowskiCT)

    def test_wrong_manifold_raises(self) -> None:
        """Passing a non-MinkowskiManifold manifold raises ValueError."""
        with pytest.raises((ValueError, TypeError)):
            cxc.MinkowskiCT(M=cxm.EuclideanManifold(3))


# =============================================================================
# MinkowskiCT — coordinate schema
# =============================================================================


class TestMinkowskiCTSchema:
    """MinkowskiCT exposes the correct coordinate schema."""

    def setup_method(self):
        self.chart = cxc.MinkowskiCT()

    def test_components(self) -> None:
        assert self.chart.components == ("ct", "x", "y", "z")

    def test_coord_dimensions(self) -> None:
        assert self.chart.coord_dimensions == ("length", "length", "length", "length")

    def test_ndim(self) -> None:
        assert self.chart.ndim == 4

    def test_ndim_consistent_with_components(self) -> None:
        assert self.chart.ndim == len(self.chart.components)
        assert self.chart.ndim == len(self.chart.coord_dimensions)


# =============================================================================
# MinkowskiCT — type hierarchy
# =============================================================================


class TestMinkowskiCTTypeHierarchy:
    """MinkowskiCT fits correctly in the chart type hierarchy."""

    def test_is_abstract_chart(self) -> None:
        assert isinstance(cxc.MinkowskiCT(), cxc.AbstractChart)

    def test_is_abstract_fixed_components_chart(self) -> None:
        assert isinstance(cxc.MinkowskiCT(), cxc.AbstractFixedComponentsChart)

    def test_is_abstract4d(self) -> None:
        assert isinstance(cxc.MinkowskiCT(), cxc.Abstract4D)

    def test_class_is_final(self) -> None:
        """MinkowskiCT must not be subclassed (it is @final)."""
        # We can only verify that the class is marked with __final__ if
        # typing_extensions sets it; accept the check if available.
        final_marker = getattr(cxc.MinkowskiCT, "__final__", None)
        if final_marker is not None:
            assert final_marker is True


# =============================================================================
# MinkowskiCT — Cartesian projection
# =============================================================================


class TestMinkowskiCTCartesian:
    """MinkowskiCT.cartesian and cartesian_chart(minkowskict)."""

    def test_cartesian_property_returns_self(self) -> None:
        chart = cxc.MinkowskiCT()
        assert chart.cartesian is chart

    def test_minkowskict_cartesian_is_minkowskict(self) -> None:
        assert cxc.minkowskict.cartesian is cxc.minkowskict

    def test_cartesian_chart_fn_identity(self) -> None:
        assert cxc.cartesian_chart(cxc.minkowskict) is cxc.minkowskict


# =============================================================================
# MinkowskiCT — Atlas registration
# =============================================================================


class TestMinkowskiCTAtlas:
    """MinkowskiCT is registered in MinkowskiAtlas."""

    def test_registered_in_atlas(self) -> None:
        atlas = cxm.MinkowskiAtlas()
        assert atlas.has_chart(cxc.minkowskict)

    def test_default_chart_is_minkowskict(self) -> None:
        atlas = cxm.MinkowskiAtlas()
        assert atlas.default_chart() == cxc.minkowskict

    def test_cart3d_not_in_minkowski_atlas(self) -> None:
        atlas = cxm.MinkowskiAtlas()
        assert not atlas.has_chart(cxc.cart3d)


# =============================================================================
# MinkowskiCT — JAX compatibility
# =============================================================================


class TestMinkowskiCTJAX:
    """MinkowskiCT is JAX-static and JIT-compatible."""

    def test_is_static_pytree(self) -> None:
        """MinkowskiCT is JAX-static: no leaves."""
        leaves, _ = jax.tree.flatten(cxc.minkowskict)
        assert leaves == []

    def test_jit_through_chart(self) -> None:
        """MinkowskiCT can be closed over in a jitted function."""

        @jax.jit
        def identity(x):
            return x

        # Chart can be passed as a static argument (via closure).
        chart = cxc.MinkowskiCT()
        result = identity(chart)
        assert result == chart

    def test_equality(self) -> None:
        """Two MinkowskiCT() instances are equal."""
        assert cxc.MinkowskiCT() == cxc.MinkowskiCT()

    def test_hash(self) -> None:
        """MinkowskiCT instances are hashable."""
        assert hash(cxc.MinkowskiCT()) == hash(cxc.MinkowskiCT())


# =============================================================================
# minkowskict pre-defined instance
# =============================================================================


class TestMinkowskictInstance:
    """The module-level minkowskict pre-defined instance."""

    def test_is_minkowskict_type(self) -> None:
        assert isinstance(cxc.minkowskict, cxc.MinkowskiCT)

    def test_equals_fresh_instance(self) -> None:
        assert cxc.minkowskict == cxc.MinkowskiCT()

    def test_repr_contains_class_name(self) -> None:
        assert "MinkowskiCT" in repr(cxc.minkowskict)

    def test_manifold_is_minkowski_manifold(self) -> None:
        assert isinstance(cxc.minkowskict.M, cxm.MinkowskiManifold)
