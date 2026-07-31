"""Tests for strategy_for_annotation utility function."""

import jaxtyping
from typing import Final

import jax.numpy as jnp
import unxt as u
import unxts.hypothesis as ust
from hypothesis import given, strategies as st
from unxts.parametric import PQ

import coordinax.charts as cxc

from coordinaxs.hypothesis.utils._src.annotations.jaxtyping_utils import (
    parse_jaxtyping_annotation,
    strategy_for_annotation,
    wrap_if_not_inspectable,
)

# The canonical unit of each named dimension. Derived here rather than imported
# from ``annotations.strategy`` so the assertion below pins the contract instead
# of comparing the implementation against itself.
CANONICAL_UNITS: Final = {
    u.unit(u.dimension(name)._unit)  # ty: ignore[unresolved-attribute]
    for name in ust.DIMENSION_NAMES
}


class TestAnnotationProcessing:
    """Test how annotations are processed in build_init_kwargs_strategy."""

    def test_jaxtype_array_detection(self) -> None:
        """Test that AbstractArray is correctly detected."""
        # Test that Shaped annotations are subclasses of AbstractArray
        ann = jaxtyping.Shaped[u.Q["length"], ""]
        assert issubclass(ann, jaxtyping.AbstractArray)

    def test_shaped_quantity_extraction(self) -> None:
        """Test extracting array_type and Metadata from Shaped[Quantity, ...]."""
        ann = jaxtyping.Shaped[u.Q["length"], ""]

        # This is what strategy_for_annotation does
        assert issubclass(ann, jaxtyping.AbstractArray)
        typ = ann.array_type
        meta = parse_jaxtyping_annotation(ann)

        # Verify the extraction
        assert typ is u.Q["length"]
        assert isinstance(meta, dict)

        # Metadata contains strategies, not raw values
        assert isinstance(meta["dtype"], st.SearchStrategy)
        assert isinstance(meta["shape"], st.SearchStrategy)

    def test_shaped_array_with_shape(self) -> None:
        """Test extracting Metadata from Shaped[Array, '3']."""
        ann = jaxtyping.Shaped[jnp.ndarray, "3"]

        assert issubclass(ann, jaxtyping.AbstractArray)
        typ = ann.array_type
        meta = parse_jaxtyping_annotation(ann)

        assert typ is jnp.ndarray
        assert isinstance(meta, dict)

    def test_non_jaxtype_annotation(self) -> None:
        """Test that non-JaxType annotations are not special-cased."""
        ann = cxc.AbstractChart

        # Regular types should not be subclasses of AbstractArray
        # (unless they happen to be jaxtyping-annotated)
        try:
            is_jaxtype = issubclass(ann, jaxtyping.AbstractArray)
        except TypeError:
            # If ann is not a class, issubclass raises TypeError
            is_jaxtype = False

        # For non-JaxType annotations, wrap_if_not_inspectable returns as-is
        if not is_jaxtype:
            wrapped = wrap_if_not_inspectable(ann)
            assert wrapped is cxc.AbstractChart


class TestStrategyForAnnotation:
    """Test strategy_for_annotation function with different argument combinations."""

    @given(st.data())
    def test_type_base_case(self, data: st.DataObject) -> None:
        """Test strategy_for_annotation(type, meta={}) - base case dispatch."""
        # When meta is empty, should use st.from_type
        # Use a concrete type (not abstract/generic) so st.from_type can resolve
        strategy = strategy_for_annotation(int, meta={})
        assert strategy is not None
        value = data.draw(strategy)
        assert isinstance(value, int)

    @given(st.data())
    def test_quantity_type_with_metadata(self, data: st.DataObject) -> None:
        """Test strategy_for_annotation(Quantity, meta) - quantity dispatch."""
        # Create Metadata from a Shaped annotation. The default Quantity is not
        # parametrized by physical type, so dimension-carrying annotations use
        # unxts.parametric.ParametricQuantity.
        ann = jaxtyping.Shaped[PQ["length"], ""]
        meta = parse_jaxtyping_annotation(ann)

        # Call with Quantity type and Metadata
        strategy = strategy_for_annotation(PQ["length"], meta=meta)
        value = data.draw(strategy)

        assert isinstance(value, u.Q)
        assert value.shape == ()
        assert u.dimension_of(value) == u.dimension("length")

    @given(st.data())
    def test_array_type_with_metadata(self, data: st.DataObject) -> None:
        """Test strategy_for_annotation(Array, meta) - array dispatch."""
        ann = jaxtyping.Shaped[jnp.ndarray, "3"]
        meta = parse_jaxtyping_annotation(ann)

        strategy = strategy_for_annotation(jnp.ndarray, meta=meta)
        value = data.draw(strategy)

        assert isinstance(value, jnp.ndarray)
        assert value.shape == (3,)

    @given(st.data())
    def test_shaped_quantity_empty_shape(self, data: st.DataObject) -> None:
        """Test Shaped[Quantity['length'], ''] produces scalar."""
        ann = jaxtyping.Shaped[u.Q["length"], ""]

        # Use wrap_if_not_inspectable and parse_jaxtyping_annotation
        wrapped = wrap_if_not_inspectable(ann)
        meta = parse_jaxtyping_annotation(ann)

        strategy = strategy_for_annotation(wrapped, meta=meta)
        value = data.draw(strategy)

        assert isinstance(value, u.Q)
        assert value.shape == ()

    @given(st.data())
    def test_shaped_quantity_with_dimension(self, data: st.DataObject) -> None:
        """Test Shaped[Quantity[Dimension(...)], ''] works."""
        ann = jaxtyping.Shaped[PQ[u.dimension("length")], ""]

        wrapped = wrap_if_not_inspectable(ann)
        meta = parse_jaxtyping_annotation(ann)

        strategy = strategy_for_annotation(wrapped, meta=meta)
        value = data.draw(strategy)

        assert isinstance(value, u.Q)
        assert value.shape == ()
        assert u.dimension_of(value) == u.dimension("length")

    @given(st.data())
    def test_undimensioned_quantity_uses_canonical_units(
        self, data: st.DataObject
    ) -> None:
        """Test an annotation pinning no dimension draws canonical units only.

        Deriving units with ``unxts.hypothesis.units()`` re-runs astropy's
        ``UnitBase.compose()`` on every draw. That is uncached and costs ~0.25s
        for exotic dimensions (e.g. "molar heat capacity"), which was enough on
        its own to trip ``HealthCheck.too_slow`` in any strategy reaching this
        fallback -- notably ``charts(filter=cxc.Abstract3D)``, via the
        un-dimensioned ``ProlateSpheroidal3D.Delta``. Sampling one canonical unit
        per named dimension avoids composing anything.
        """
        ann = jaxtyping.Real[u.quantity.StaticQuantity, ""]
        wrapped = wrap_if_not_inspectable(ann)
        meta = parse_jaxtyping_annotation(ann)

        value = data.draw(strategy_for_annotation(wrapped, meta=meta))

        assert isinstance(value, u.quantity.StaticQuantity)
        assert value.unit in CANONICAL_UNITS
