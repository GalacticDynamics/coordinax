"""`coordinaxs.hypothesis.utils.draw_if_strategy`: draw, or pass through."""

__all__: tuple[str, ...] = ()

import hypothesis.strategies as st
import pytest
from hypothesis import given

from coordinaxs.hypothesis.utils import draw_if_strategy

#: Non-strategy values must come back by identity, not be iterated or coerced.
PLAIN_VALUES = [
    pytest.param(42, id="int"),
    pytest.param(None, id="none"),
    pytest.param("hello", id="str"),
    pytest.param([1, 2, 3], id="list"),
    pytest.param({"key": "val"}, id="dict"),
]


@pytest.mark.parametrize("value", PLAIN_VALUES)
@given(data=st.data())
def test_plain_value_returned_unchanged(value: object, data: st.DataObject) -> None:
    assert draw_if_strategy(data.draw, value) is value


@given(data=st.data())
def test_draws_from_a_strategy(data: st.DataObject) -> None:
    result = draw_if_strategy(data.draw, st.integers(min_value=0, max_value=100))
    assert isinstance(result, int)
    assert 0 <= result <= 100


@given(data=st.data())
def test_draws_from_just(data: st.DataObject) -> None:
    """`st.just(x)` is a strategy, so it is drawn from rather than returned."""
    sentinel = object()
    assert draw_if_strategy(data.draw, st.just(sentinel)) is sentinel


@given(data=st.data())
def test_draws_from_sampled_from(data: st.DataObject) -> None:
    options = ("a", "b", "c")
    assert draw_if_strategy(data.draw, st.sampled_from(options)) in options
