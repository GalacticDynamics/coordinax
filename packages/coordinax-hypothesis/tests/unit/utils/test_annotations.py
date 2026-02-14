"""Tests for the ``coordinax_hypothesis.utils.annotations`` public API."""

import jaxtyping
from typing import Annotated

import hypothesis.strategies as st
import jax.numpy as jnp
import pytest
from beartype.vale import Is
from hypothesis import given

import unxt as u

from coordinax_hypothesis.utils import annotations

# =========================================================================
# Metadata
# =========================================================================


class TestMetadata:
    """Tests for the Metadata TypedDict."""

    def test_empty_metadata(self):
        meta = annotations.Metadata()
        assert isinstance(meta, dict)
        assert len(meta) == 0

    def test_metadata_with_dtype(self):
        meta = annotations.Metadata(dtype=jnp.float32)
        assert meta["dtype"] is jnp.float32

    def test_metadata_with_shape(self):
        meta = annotations.Metadata(shape=())
        assert meta["shape"] == ()

    def test_metadata_with_validators(self):
        validators = [lambda x: x > 0]
        meta = annotations.Metadata(validators=validators)
        assert meta["validators"] is validators

    def test_metadata_with_all_keys(self):
        validators = [lambda _: True]
        meta = annotations.Metadata(
            dtype=jnp.float64, shape=(3,), validators=validators
        )
        assert meta["dtype"] is jnp.float64
        assert meta["shape"] == (3,)
        assert meta["validators"] is validators

    def test_metadata_is_dict(self):
        """Metadata is a TypedDict and therefore a plain dict."""
        meta = annotations.Metadata(dtype=jnp.float32)
        assert isinstance(meta, dict)


# =========================================================================
# AbstractNotIntrospectable / wrappers
# =========================================================================


class TestAbstractNotIntrospectable:
    """Tests for wrapper dataclasses."""

    def test_abstract_wrapper_stores_annotation(self):
        wrapper = annotations.AbstractNotIntrospectable(ann=int)
        assert wrapper.ann is int

    def test_abstract_wrapper_frozen(self):
        wrapper = annotations.AbstractNotIntrospectable(ann=int)
        with pytest.raises(AttributeError):
            wrapper.ann = float  # type: ignore[misc]

    def test_annotated_wrapper_is_subclass(self):
        assert issubclass(
            annotations.AnnotatedNotIntrospectable,
            annotations.AbstractNotIntrospectable,
        )

    def test_annotated_wrapper_stores_annotation(self):
        ann = Annotated[int, "extra"]
        wrapper = annotations.AnnotatedNotIntrospectable(ann=ann)
        assert wrapper.ann is ann

    def test_jaxtyping_wrapper_is_subclass(self):
        assert issubclass(
            annotations.JaxtypingNotIntrospectable,
            annotations.AbstractNotIntrospectable,
        )

    def test_jaxtyping_wrapper_stores_annotation(self):
        ann = jaxtyping.Shaped[jnp.ndarray, "3"]
        wrapper = annotations.JaxtypingNotIntrospectable(ann=ann)
        assert wrapper.ann is ann


# =========================================================================
# RECOGNIZE_NONINTROSPECTABLE registry
# =========================================================================


class TestRecognizeNonintrospectable:
    """Tests for the RECOGNIZE_NONINTROSPECTABLE registry."""

    def test_is_list(self):
        assert isinstance(annotations.RECOGNIZE_NONINTROSPECTABLE, list)

    def test_has_entries(self):
        assert (
            len(annotations.RECOGNIZE_NONINTROSPECTABLE) >= 2
        )  # Annotated + jaxtyping

    def test_entries_are_tuples(self):
        for entry in annotations.RECOGNIZE_NONINTROSPECTABLE:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            check_fn, wrapper_cls = entry
            assert callable(check_fn)
            assert issubclass(wrapper_cls, annotations.AbstractNotIntrospectable)

    def test_annotated_entry_present(self):
        """An entry for Annotated types should use AnnotatedNotIntrospectable."""
        wrapper_classes = [cls for _, cls in annotations.RECOGNIZE_NONINTROSPECTABLE]
        assert annotations.AnnotatedNotIntrospectable in wrapper_classes

    def test_jaxtyping_entry_present(self):
        """An entry for jaxtyping types should use JaxtypingNotIntrospectable."""
        wrapper_classes = [cls for _, cls in annotations.RECOGNIZE_NONINTROSPECTABLE]
        assert annotations.JaxtypingNotIntrospectable in wrapper_classes


# =========================================================================
# wrap_if_not_inspectable
# =========================================================================


class TestWrapIfNotInspectable:
    """Tests for wrap_if_not_inspectable."""

    def test_plain_type_passthrough(self):
        """A plain type is not wrapped."""
        result = annotations.wrap_if_not_inspectable(int)
        assert result is int

    def test_plain_class_passthrough(self):
        class Foo:
            pass

        result = annotations.wrap_if_not_inspectable(Foo)
        assert result is Foo

    def test_annotated_type_wrapped(self):
        ann = Annotated[int, "extra"]
        result = annotations.wrap_if_not_inspectable(ann)
        assert isinstance(result, annotations.AnnotatedNotIntrospectable)
        assert result.ann is ann

    def test_jaxtyping_type_wrapped(self):
        ann = jaxtyping.Shaped[jnp.ndarray, "3"]
        result = annotations.wrap_if_not_inspectable(ann)
        assert isinstance(result, annotations.JaxtypingNotIntrospectable)
        assert result.ann is ann

    def test_annotated_quantity_wrapped(self):
        ann = Annotated[u.Q["length"], Is[lambda x: x.value > 0]]
        result = annotations.wrap_if_not_inspectable(ann)
        assert isinstance(result, annotations.AnnotatedNotIntrospectable)

    def test_jaxtyping_quantity_wrapped(self):
        ann = jaxtyping.Shaped[u.Q["length"], ""]
        result = annotations.wrap_if_not_inspectable(ann)
        assert isinstance(result, annotations.JaxtypingNotIntrospectable)


# =========================================================================
# strategy_for_annotation
# =========================================================================


class TestStrategyForAnnotation:
    """Tests for the dispatched strategy_for_annotation function."""

    # --- base type dispatch ---

    @given(st.data())
    def test_plain_type_generates_instance(self, data: st.DataObject):
        """Base dispatch: st.from_type for a plain type."""
        strategy = annotations.strategy_for_annotation(int, meta=annotations.Metadata())
        value = data.draw(strategy)
        assert isinstance(value, int)

    # --- jax.Array dispatch ---

    @given(st.data())
    def test_jax_array_dispatch(self, data: st.DataObject):
        strategy = annotations.strategy_for_annotation(
            jnp.ndarray, meta=annotations.Metadata(dtype=jnp.float32, shape=(2,))
        )
        value = data.draw(strategy)
        assert isinstance(value, jnp.ndarray)
        assert value.shape == (2,)
        assert value.dtype == jnp.float32

    @given(st.data())
    def test_jax_array_scalar(self, data: st.DataObject):
        strategy = annotations.strategy_for_annotation(
            jnp.ndarray, meta=annotations.Metadata(dtype=jnp.float64, shape=())
        )
        value = data.draw(strategy)
        assert isinstance(value, jnp.ndarray)
        assert value.shape == ()

    # --- Quantity dispatch ---

    @given(st.data())
    def test_quantity_dispatch(self, data: st.DataObject):
        strategy = annotations.strategy_for_annotation(
            u.Q["length"],
            meta=annotations.Metadata(dtype=jnp.float64, shape=()),
        )
        value = data.draw(strategy)
        assert isinstance(value, u.Q)
        assert u.dimension_of(value) == u.dimension("length")
        assert value.shape == ()

    @given(st.data())
    def test_quantity_dispatch_with_shape(self, data: st.DataObject):
        strategy = annotations.strategy_for_annotation(
            u.Q["length"],
            meta=annotations.Metadata(dtype=jnp.float64, shape=(3,)),
        )
        value = data.draw(strategy)
        assert isinstance(value, u.Q)
        assert value.shape == (3,)

    # --- AnnotatedNotIntrospectable dispatch ---

    @given(st.data())
    def test_annotated_dispatch(self, data: st.DataObject):
        """Annotated[Quantity, ...] is unwrapped and re-dispatched."""
        ann = Annotated[u.Q["length"], {"dtype": jnp.float64, "shape": ()}]
        wrapped = annotations.wrap_if_not_inspectable(ann)
        strategy = annotations.strategy_for_annotation(
            wrapped, meta=annotations.Metadata()
        )
        value = data.draw(strategy)
        assert isinstance(value, u.Q)
        assert u.dimension_of(value) == u.dimension("length")

    # --- JaxtypingNotIntrospectable dispatch ---

    @given(st.data())
    def test_jaxtyping_dispatch_array(self, data: st.DataObject):
        ann = jaxtyping.Shaped[jnp.ndarray, "3"]
        wrapped = annotations.wrap_if_not_inspectable(ann)
        strategy = annotations.strategy_for_annotation(
            wrapped, meta=annotations.Metadata()
        )
        value = data.draw(strategy)
        assert isinstance(value, jnp.ndarray)
        assert value.shape == (3,)

    @given(st.data())
    def test_jaxtyping_dispatch_scalar(self, data: st.DataObject):
        ann = jaxtyping.Shaped[jnp.ndarray, ""]
        wrapped = annotations.wrap_if_not_inspectable(ann)
        strategy = annotations.strategy_for_annotation(
            wrapped, meta=annotations.Metadata()
        )
        value = data.draw(strategy)
        assert isinstance(value, jnp.ndarray)
        assert value.shape == ()

    @given(st.data())
    def test_jaxtyping_dispatch_quantity(self, data: st.DataObject):
        ann = jaxtyping.Shaped[u.Q["length"], ""]
        wrapped = annotations.wrap_if_not_inspectable(ann)
        strategy = annotations.strategy_for_annotation(
            wrapped, meta=annotations.Metadata()
        )
        value = data.draw(strategy)
        assert isinstance(value, u.Q)
        assert value.shape == ()

    @given(st.data())
    def test_jaxtyping_float_dtype(self, data: st.DataObject):
        ann = jaxtyping.Float[jnp.ndarray, ""]
        wrapped = annotations.wrap_if_not_inspectable(ann)
        strategy = annotations.strategy_for_annotation(
            wrapped, meta=annotations.Metadata()
        )
        value = data.draw(strategy)
        assert isinstance(value, jnp.ndarray)
        assert jnp.issubdtype(value.dtype, jnp.floating)

    @given(st.data())
    def test_jaxtyping_integer_dtype(self, data: st.DataObject):
        ann = jaxtyping.Integer[jnp.ndarray, ""]
        wrapped = annotations.wrap_if_not_inspectable(ann)
        strategy = annotations.strategy_for_annotation(
            wrapped, meta=annotations.Metadata()
        )
        value = data.draw(strategy)
        assert isinstance(value, jnp.ndarray)
        assert jnp.issubdtype(value.dtype, jnp.integer)

    # --- meta merging ---

    @given(st.data())
    def test_meta_override_in_annotated(self, data: st.DataObject):
        """Metadata embedded in Annotated overrides the default."""
        ann = Annotated[u.Q["length"], {"dtype": jnp.float32, "shape": (5,)}]
        wrapped = annotations.wrap_if_not_inspectable(ann)
        strategy = annotations.strategy_for_annotation(
            wrapped, meta=annotations.Metadata()
        )
        value = data.draw(strategy)
        assert isinstance(value, u.Q)
        assert value.shape == (5,)


# =========================================================================
# cached_strategy_for_annotation
# =========================================================================


class TestCachedStrategyForAnnotation:
    """Tests for cached_strategy_for_annotation."""

    def test_returns_strategy(self):
        result = annotations.cached_strategy_for_annotation(int)
        assert isinstance(result, st.SearchStrategy)

    def test_caching_identity(self):
        """Same input returns the same strategy object (cached)."""
        s1 = annotations.cached_strategy_for_annotation(int)
        s2 = annotations.cached_strategy_for_annotation(int)
        assert s1 is s2

    @given(st.data())
    def test_cached_jax_array(self, data: st.DataObject):
        strategy = annotations.cached_strategy_for_annotation(jnp.ndarray)
        value = data.draw(strategy)
        assert isinstance(value, jnp.ndarray)

    @given(st.data())
    def test_cached_quantity(self, data: st.DataObject):
        strategy = annotations.cached_strategy_for_annotation(u.Q["length"])
        value = data.draw(strategy)
        assert isinstance(value, u.Q)

    @given(st.data())
    def test_cached_jaxtyping_wrapped(self, data: st.DataObject):
        ann = jaxtyping.Shaped[jnp.ndarray, "3"]
        wrapped = annotations.wrap_if_not_inspectable(ann)
        strategy = annotations.cached_strategy_for_annotation(wrapped)
        value = data.draw(strategy)
        assert isinstance(value, jnp.ndarray)
        assert value.shape == (3,)
