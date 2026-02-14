"""Tests for strategy_for_annotation utility function."""

import jaxtyping

import jax.numpy as jnp
from hypothesis import given, strategies as st

import unxt as u

import coordinax.charts as cxc
from coordinax_hypothesis.utils._src.annotations.jaxtyping_utils import (
    parse_jaxtyping_annotation,
    strategy_for_annotation,
    wrap_if_not_inspectable,
)


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
        # Create Metadata from a Shaped annotation
        ann = jaxtyping.Shaped[u.Q["length"], ""]
        meta = parse_jaxtyping_annotation(ann)

        # Call with Quantity type and Metadata
        strategy = strategy_for_annotation(u.Q["length"], meta=meta)
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
        ann = jaxtyping.Shaped[u.Q[u.dimension("length")], ""]

        wrapped = wrap_if_not_inspectable(ann)
        meta = parse_jaxtyping_annotation(ann)

        strategy = strategy_for_annotation(wrapped, meta=meta)
        value = data.draw(strategy)

        assert isinstance(value, u.Q)
        assert value.shape == ()
        assert u.dimension_of(value) == u.dimension("length")
