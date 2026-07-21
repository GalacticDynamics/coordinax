"""Tests for the public ``make_nonnegative`` helper.

``make_nonnegative`` is part of the public API of
``coordinaxs.hypothesis.distances`` (it is the shared non-negativity helper
behind ``distances`` and the astro parallax strategies), so it is imported here
via that public path.
"""

import hypothesis.strategies as st
import jax.numpy as jnp
from hypothesis import given

from coordinaxs.hypothesis.distances import make_nonnegative


def test_public_import() -> None:
    """The helper is importable from the package public API and callable."""
    from coordinaxs.hypothesis import distances as cxdst

    assert "make_nonnegative" in cxdst.__all__
    assert cxdst.make_nonnegative is make_nonnegative
    assert callable(make_nonnegative)


@given(data=st.data())
def test_mapping_elements_raises_min_value(data: st.DataObject) -> None:
    """A mapping ``elements`` has its ``min_value`` raised to at least 0."""
    out = make_nonnegative(data.draw, elements={"min_value": -10, "max_value": 100})
    assert out["elements"]["min_value"] == 0
    assert out["elements"]["max_value"] == 100  # other keys untouched


@given(data=st.data())
def test_mapping_elements_keeps_nonnegative_min(data: st.DataObject) -> None:
    """A mapping ``elements`` with a non-negative ``min_value`` is preserved."""
    out = make_nonnegative(data.draw, elements={"min_value": 5.0})
    assert out["elements"]["min_value"] == 5.0


@given(data=st.data())
def test_strategy_elements_are_made_nonnegative(data: st.DataObject) -> None:
    """A ``SearchStrategy`` ``elements`` yields only non-negative draws."""
    out = make_nonnegative(data.draw, elements=st.floats(-100.0, 100.0))
    value = data.draw(out["elements"])
    assert value >= 0


@given(data=st.data())
def test_absent_elements_creates_nonnegative_default(data: st.DataObject) -> None:
    """With no ``elements``, a default non-negative strategy is created."""
    out = make_nonnegative(data.draw, dtype=jnp.float32)
    assert "elements" in out
    value = data.draw(out["elements"])
    assert value >= 0


@given(data=st.data())
def test_other_kwargs_are_passed_through(data: st.DataObject) -> None:
    """Keys other than ``elements`` are returned unchanged."""
    out = make_nonnegative(data.draw, unit="kpc", shape=(2, 3), elements={})
    assert out["unit"] == "kpc"
    assert out["shape"] == (2, 3)
