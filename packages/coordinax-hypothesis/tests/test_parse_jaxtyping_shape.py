"""Tests for parse_jaxtyping_shape utility function."""

import jaxtyping

import jax.numpy as jnp
from hypothesis import given, strategies as st

from coordinax_hypothesis._src.utils import (
    _is_variadic_dim,
    _make_size_strategy,
    parse_jaxtyping_shape,
)


class TestIsVariadicDim:
    """Test _is_variadic_dim helper."""

    def test_ellipsis_is_variadic(self) -> None:
        """Test that Ellipsis is recognized as variadic."""
        ann = jaxtyping.Shaped[jnp.ndarray, "..."]
        dim = ann.dims[0]
        # Ellipsis represented as object, but we check identity
        assert _is_variadic_dim(dim)

    def test_named_variadic_is_variadic(self) -> None:
        """Test that *batch is recognized as variadic."""
        ann = jaxtyping.Shaped[jnp.ndarray, "*batch"]
        dim = ann.dims[0]
        assert _is_variadic_dim(dim)

    def test_fixed_dim_not_variadic(self) -> None:
        """Test that fixed dimensions are not variadic."""
        ann = jaxtyping.Shaped[jnp.ndarray, "3"]
        dim = ann.dims[0]
        assert not _is_variadic_dim(dim)

    def test_named_dim_not_variadic(self) -> None:
        """Test that named dimensions are not variadic."""
        ann = jaxtyping.Shaped[jnp.ndarray, "n"]
        dim = ann.dims[0]
        assert not _is_variadic_dim(dim)


class TestMakeSizeStrategy:
    """Test _make_size_strategy helper."""

    @given(st.data())
    def test_fixed_dim_as_int(self, data: st.DataObject) -> None:
        """Test strategy for fixed dimension returns int."""
        ann = jaxtyping.Shaped[jnp.ndarray, "3"]
        dim = ann.dims[0]

        strategy = _make_size_strategy(dim, as_list=False)
        value = data.draw(strategy)

        assert isinstance(value, int)
        assert value == 3

    @given(st.data())
    def test_fixed_dim_as_list(self, data: st.DataObject) -> None:
        """Test strategy for fixed dimension can return list."""
        ann = jaxtyping.Shaped[jnp.ndarray, "3"]
        dim = ann.dims[0]

        strategy = _make_size_strategy(dim, as_list=True)
        value = data.draw(strategy)

        assert isinstance(value, list)
        assert value == [3]

    @given(st.data())
    def test_broadcastable_fixed_dim(self, data: st.DataObject) -> None:
        """Test strategy for broadcastable fixed dimension."""
        ann = jaxtyping.Shaped[jnp.ndarray, "#3"]
        dim = ann.dims[0]

        strategy = _make_size_strategy(dim, as_list=False)
        value = data.draw(strategy)

        assert isinstance(value, int)
        assert value in [1, 3]

    @given(st.data())
    def test_named_dim(self, data: st.DataObject) -> None:
        """Test strategy for named dimension generates variable size."""
        ann = jaxtyping.Shaped[jnp.ndarray, "n"]
        dim = ann.dims[0]

        strategy = _make_size_strategy(dim, as_list=False)
        value = data.draw(strategy)

        assert isinstance(value, int)
        assert 1 <= value <= 10

    @given(st.data())
    def test_broadcastable_named_dim(self, data: st.DataObject) -> None:
        """Test strategy for broadcastable named dimension."""
        ann = jaxtyping.Shaped[jnp.ndarray, "#n"]
        dim = ann.dims[0]

        strategy = _make_size_strategy(dim, as_list=False)
        value = data.draw(strategy)

        assert isinstance(value, int)
        assert 1 <= value <= 10  # Can be 1 (broadcasted) or 1-10 (variable)

    @given(st.data())
    def test_variadic_dim(self, data: st.DataObject) -> None:
        """Test strategy for variadic dimension generates list."""
        ann = jaxtyping.Shaped[jnp.ndarray, "*batch"]
        dim = ann.dims[0]

        strategy = _make_size_strategy(dim)  # Always returns list
        value = data.draw(strategy)

        assert isinstance(value, list)
        assert 0 <= len(value) <= 3
        assert all(isinstance(v, int) and 1 <= v <= 10 for v in value)


class TestParseJaxTypingShape:
    """Test parse_jaxtyping_shape main function."""

    @given(st.data())
    def test_scalar_shape(self, data: st.DataObject) -> None:
        """Test parsing scalar (empty) shape."""
        ann = jaxtyping.Shaped[jnp.ndarray, ""]
        strategy = parse_jaxtyping_shape(ann.dims)
        value = data.draw(strategy)

        assert value == ()

    @given(st.data())
    def test_fixed_shape(self, data: st.DataObject) -> None:
        """Test parsing fixed shape."""
        ann = jaxtyping.Shaped[jnp.ndarray, "3 4"]
        strategy = parse_jaxtyping_shape(ann.dims)
        value = data.draw(strategy)

        assert value == (3, 4)

    @given(st.data())
    def test_named_shape(self, data: st.DataObject) -> None:
        """Test parsing named variable shape."""
        ann = jaxtyping.Shaped[jnp.ndarray, "batch channels"]
        strategy = parse_jaxtyping_shape(ann.dims)
        value = data.draw(strategy)

        assert len(value) == 2
        assert all(isinstance(v, int) and 1 <= v <= 10 for v in value)

    @given(st.data())
    def test_mixed_fixed_and_named(self, data: st.DataObject) -> None:
        """Test parsing mix of fixed and named dimensions."""
        ann = jaxtyping.Shaped[jnp.ndarray, "3 channels 4"]
        strategy = parse_jaxtyping_shape(ann.dims)
        value = data.draw(strategy)

        assert len(value) == 3
        assert value[0] == 3
        assert isinstance(value[1], int)
        assert 1 <= value[1] <= 10
        assert value[2] == 4

    @given(st.data())
    def test_variadic_with_fixed(self, data: st.DataObject) -> None:
        """Test parsing variadic with fixed dimensions."""
        ann = jaxtyping.Shaped[jnp.ndarray, "*batch 3"]
        strategy = parse_jaxtyping_shape(ann.dims)
        value = data.draw(strategy)

        # Last element should be 3, variadic can add 0-3 dims before it
        assert value[-1] == 3
        assert 1 <= len(value) <= 4  # 0-3 variadic + 1 fixed

    @given(st.data())
    def test_ellipsis_with_fixed(self, data: st.DataObject) -> None:
        """Test parsing ellipsis with fixed dimensions."""
        ann = jaxtyping.Shaped[jnp.ndarray, "... 3"]
        strategy = parse_jaxtyping_shape(ann.dims)
        value = data.draw(strategy)

        # Last element should be 3, ellipsis can add 0-3 dims before it
        assert value[-1] == 3
        assert 1 <= len(value) <= 4

    @given(st.data())
    def test_broadcastable_dimensions(self, data: st.DataObject) -> None:
        """Test parsing broadcastable dimensions."""
        ann = jaxtyping.Shaped[jnp.ndarray, "#3 #n"]
        strategy = parse_jaxtyping_shape(ann.dims)
        value = data.draw(strategy)

        assert len(value) == 2
        assert value[0] in [1, 3]
        assert isinstance(value[1], int)
        assert 1 <= value[1] <= 10
